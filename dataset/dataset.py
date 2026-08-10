"""Streaming dataset construction for pretraining and supervised fine-tuning.

Adapted from Open-Llama/dataset/dataset.py.
"""
import copy
import math
import random
from glob import glob

import torch
from datasets import load_dataset

from models.tokenization_vocab_32k_gpt2 import Vocab32kGPT2Tokenizer

random.seed(42)


def pretrain_transform(batch):
    """Normalize a raw pretraining record into a single `text` field.

    SkyPile-150B and OpenWebText already ship a `text` field, so this is a no-op. Corpora with a different schema
    should be folded into `text` here before tokenization.
    """
    if ("text") in batch:
        pass
    return batch


def _prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}</s>").format_map(row)


def _prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}</s>").format_map(row)

def prompt_no_input_no_history(row):
    return ("### Instruction:\n{instruction}\n\n### System:\n{output}</s>").format_map(row)

def prompt_input(row):
    return ("### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### System:\n{output}</s>").format_map(row)


def instruct_transform(batch):
    answer = []
    if batch["input"] != ['']:
        text = prompt_input({'instruction':batch["instruction"][0],'input':batch["input"][0], 'output':batch["output"][0]})
        answer.append(batch["output"][0] + '</s>')

    elif batch["history"][0]:
        chats = []
        for u, r in batch["history"][0]:
            chats.append(prompt_no_input_no_history({'instruction':u,'output':r}))
            answer.append(r + '</s>')
        chats.append(prompt_no_input_no_history({'instruction':batch["instruction"][0],'output':batch["output"][0]})) #current
        answer.append(batch["output"][0] + '</s>')
        text = '[multiturn_sep]'.join(chats)
        
    else:
        text = prompt_no_input_no_history({'instruction':batch["instruction"][0],'output':batch["output"][0]})
        answer.append(batch["output"][0] + '</s>')
    
    answer = '[multiturn_sep]'.join(answer)

    return {"text": [text], "answer": [answer]}

def split_multiturn(batch):
    return {"text": batch["text"][0].split("[multiturn_sep]"), "answer": batch["answer"][0].split("[multiturn_sep]")}


def sample_sequence_gen(seq_length, eos_token_id):
    def sample_sequence(line):
        doc_length = line["input_ids"].shape[0]
        if doc_length <= seq_length:
            start = 0
        else:
            if random.random() < 1 / 4:
                start = 0
            else:
                start = random.randint(0, doc_length - seq_length)
        input_ids = line["input_ids"][start : start + seq_length]
        if input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        return {"input_ids": input_ids}

    return sample_sequence


def split_sequence_gen(seq_length):
    def split_sequence(batch):
        input_ids = batch["input_ids"][0]
        out = []
        while len(input_ids) >= (1 + len(out)) * seq_length:
            out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
        return {"input_ids": out}

    return split_sequence


def concat_multiple_sequence_gen(seq_length, pad_id):
    def concat_multiple_sequence(batch):
        concat_input_ids = torch.cat(batch["input_ids"], dim=0)
        length = concat_input_ids.shape[0]
        chunks = math.ceil(length / seq_length)
        pad_length = chunks * seq_length - length
        pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_id
        concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)
        input_ids = torch.chunk(concat_input_ids, chunks)
        return {"input_ids": input_ids}

    return concat_multiple_sequence

def get_labels_gen(pad_id):
    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        return {"labels": labels}

    return get_labels

def get_sft_labels_gen(example, pad_id):
    labels = example["labels"]
    labels[labels == pad_id] = -100
    return {"labels": labels}

def mask_process(example, tokenizer, seq_length):

    all_text = example["text"]
    target = example["answer"]
    source = example["text"][:len(all_text) - len(target)]


    all_text_tokenized = tokenizer(
        all_text,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )

    all_input_ids = all_text_tokenized.input_ids[0]

    source_tokenized = tokenizer(
        source,
        return_tensors="pt",
        return_attention_mask=False,
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )

    source_len = source_tokenized["input_ids"][0].ne(tokenizer.pad_id).sum().item()

    input_ids = all_input_ids
    labels = copy.deepcopy(input_ids)
    labels[:source_len] = -100

    return {"input_ids":input_ids, "labels":labels, "attention_mask":all_text_tokenized['attention_mask'][0]}


def construct_dataset(
    dataset_config, tokenizer, return_raw_text=False, world_size=None
):
    all_data_files = []
    for name, pattern in dataset_config["data"].items():
        data_files = glob(pattern)
        assert len(data_files) > 0
        all_data_files.extend(data_files)
    random.shuffle(all_data_files)
    # When the shard count is divisible by world_size, split_dataset_by_node shards the files directly. Otherwise it
    # reads every record and skips part of them on each rank, which is noticeably slower.
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.distributed.split_dataset_by_node
    if world_size is not None:
        num_shards = len(all_data_files)
        all_data_files = all_data_files[: num_shards // world_size * world_size]
    dataset = load_dataset(
        "json", data_files=all_data_files, split="train", streaming=True
    )
    # shuffle
    dataset = dataset.shuffle(seed=42)
    # Normalize the raw records into a common text schema
    if dataset_config["mode"] == "pretrain":
        dataset = dataset.map(pretrain_transform, batched=True, batch_size=1)
    elif dataset_config["mode"] == "instruct":
        dataset = dataset.map(instruct_transform, batched=True, batch_size=1)
        dataset = dataset.select_columns(["text", "answer"])
        dataset = dataset.map(split_multiturn, batched=True, batch_size=1)
    else:
        raise Exception("Dataset mode: {} not found.".format(dataset_config["mode"]))

    full_dataset = dataset

    # to visualize
    # if return_raw_text:
    #     return full_dataset

    seq_length = dataset_config["seq_length"]
    pad_to_max = dataset_config.get("pad_to_max", True)
    sequence_sample_mode = dataset_config.get("sequence_sample_mode", "truncation")
    truncation = sequence_sample_mode == "truncation"
    concat_multiple_sequence = dataset_config.get("concat_multiple_sequence", False)

    # tokenize
    if pad_to_max:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                padding="max_length",
                max_length=seq_length,
                truncation=truncation,
            )
        )
    else:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                truncation=truncation,
            )
        )

    if(dataset_config["mode"] == "instruct"):
        full_dataset = full_dataset.map(lambda example:mask_process(example, tokenizer, seq_length), batched=False)
    # format
    else:
        full_dataset = full_dataset.map(lambda x: {"input_ids": x["input_ids"][0]})
    if(dataset_config["mode"] == "instruct"):
        full_dataset = full_dataset.select_columns(["input_ids", "labels", "attention_mask"])
    else:
        full_dataset = full_dataset.select_columns("input_ids")

    # sequence_sample
    if sequence_sample_mode == "truncation":
        pass
    elif sequence_sample_mode == "none":
        pass
    elif sequence_sample_mode == "sample":
        assert pad_to_max or concat_multiple_sequence
        full_dataset = full_dataset.map(
            sample_sequence_gen(seq_length, tokenizer.eos_token_id)
        )
    elif sequence_sample_mode == "split":
        assert not concat_multiple_sequence
        full_dataset = full_dataset.map(
            split_sequence_gen(seq_length), batched=True, batch_size=1
        )
    else:
        raise Exception(
            "Unknown sequence_sample mode: {}.".format(sequence_sample_mode)
        )

    # concat multiple sequence
    if concat_multiple_sequence:
        num_sequences = dataset_config["num_sequences"]
        full_dataset = full_dataset.map(
            concat_multiple_sequence_gen(seq_length, tokenizer.pad_id),
            batched=True,
            batch_size=num_sequences,
            drop_last_batch=True,
        )

    # labels: SFT masks the prompt, pretraining trains on every non-padding token
    if(dataset_config["mode"] == "instruct"):
        full_dataset = full_dataset.map(lambda example: get_sft_labels_gen(example, tokenizer.pad_id))
    else:
        full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_id))
    # shuffle
    full_dataset = full_dataset.shuffle(seed=42)
    return full_dataset


if __name__ == "__main__":
    # Smoke test: check that tokenize -> decode round-trips, then dump one collated batch.
    import time
    from torch.utils.data import DataLoader

    data_config = {
        "mode": "instruct",
        "data": {"mixed": "test.jsonl"},
        "pad_to_max": True,
        "sequence_sample_mode": "truncation",
        "concat_multiple_sequence": False,
        "num_sequences": 10,
        "seq_length": 1000,
        "tokenizer_model_path": "configs/tokenizer_models/vocab_32k_gpt2_moe.model"
    }
    tokenizer = Vocab32kGPT2Tokenizer(vocab_file=data_config["tokenizer_model_path"], legacy=False)
    pretrain_dataset = construct_dataset(data_config, tokenizer, True)
    start = time.time()
    for i, line in enumerate(pretrain_dataset):
        raw_text = line["text"]
        input_ids = tokenizer(
            line["text"], return_tensors="pt", return_attention_mask=False
        )["input_ids"][0]
        decode_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if raw_text != decode_text and "▁" not in raw_text:
            print(raw_text, "\n", decode_text)
        if i == 10:
            break
    print("all checked in {} seconds.".format(time.time() - start))
    pretrain_dataset = construct_dataset(data_config, tokenizer)
    print(pretrain_dataset.n_shards)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, num_workers=16)
    for batch in pretrain_loader:
        for k, v in batch.items():
            print(k, v.shape, "\n", v)
        break

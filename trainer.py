"""Training loop with DeepSpeed/Accelerate, cosine schedule, W&B logging and checkpoint resumption."""

import logging
import os
import time

import torch
import wandb
from deepspeed.ops.adam import FusedAdam
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from dataset.validation import val_set_pretrain

# Log to a local W&B directory by default so that training never blocks on network access. Export
# WANDB_MODE=online together with WANDB_API_KEY to stream metrics to the cloud instead.
os.environ.setdefault("WANDB_MODE", "offline")


class Trainer:
    def __init__(self, config, raw_model, train_loader, tokenizer, accelerator):
        self.config = config
        self.raw_model = raw_model
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.train_and_eval = config["train"].get("train_and_eval", False)
        self.gradient_accumulation_steps = config["train"].get(
            "gradient_accumulation_steps", 1
        )
        self.lr_scheduler_factor = (
            accelerator.num_processes / accelerator.gradient_accumulation_steps
        )
        self.log_interval = (
            self.config["log_interval"] * accelerator.gradient_accumulation_steps
        )
        self.eval_interval = (
            self.config["eval_interval"] * accelerator.gradient_accumulation_steps
        )
        self.save_interval = (
            self.config["save_interval"] * accelerator.gradient_accumulation_steps
        )
        self.work_dir = self.config["work_dir"]
        if accelerator.is_main_process:
            wandb.init(project=self.config["project_name"])

    def get_model_info(self):
        with torch.no_grad():
            summary(
                self.raw_model.cuda(),
                input_data=torch.ones(1, 64, dtype=torch.int64).cuda(),
            )

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        if self.config["train"].get("use_lora", False):
            optimizer_grouped_parameters = self.raw_model.parameters()
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config["train"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        self.optim = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.config["train"]["lr"],
            betas=(0.9, 0.95),
        )

    def get_lr_scheduler(self):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=self.config["train"]["num_warmup_steps"]
            * self.lr_scheduler_factor,
            num_training_steps=self.config["train"]["num_training_steps"]
            * self.lr_scheduler_factor,
        )

    def prepare(self):
        (
            _,
            self.model,
            self.optim,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_loader, self.raw_model, self.optim, self.scheduler
        )
        self.optim.zero_grad()
        self.global_step = 0
        # Accelerate raises ValueError when work_dir holds no checkpoint and FileNotFoundError when the
        # checkpoint is incomplete; both simply mean there is nothing to resume from. Any other failure
        # (corrupt shards, shape mismatch, ...) must surface instead of being downgraded to a fresh run.
        try:
            self.accelerator.load_state(self.work_dir)
        except (FileNotFoundError, ValueError) as error:
            logging.warning(
                "No ckpt to resume in {}, training from scratch ({}: {})".format(
                    self.work_dir, type(error).__name__, error
                )
            )
        else:
            self.global_step = self.scheduler.scheduler._step_count - 1
            self.global_step = self.global_step // self.accelerator.num_processes
            logging.warning(
                "Restored ckpt from {} at global step {}".format(self.work_dir, self.global_step)
            )
        if self.global_step > 0:
            skip_steps = self.global_step * self.gradient_accumulation_steps
            logging.warning("Skipped {} steps.".format(skip_steps))
            self.train_loader_skipped = self.accelerator.skip_first_batches(self.train_loader, num_batches=skip_steps)
        else:
            self.train_loader_skipped = self.train_loader
        self.accelerator.wait_for_everyone()

    def train_step(self, batch):
        out = self.model(**batch)
        total_loss = out.loss
        losses = {"total_loss": total_loss}
        self.accelerator.backward(total_loss)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()
        return losses

    def train(self):
        self.get_optimizer()
        self.get_lr_scheduler()
        self.prepare()
        self.start_time = time.time()
        self.epoch = 0
        self.data_step = 0

        while True:
            if self.data_step >= self.config["train"]["num_training_steps"]:
                break
            if self.epoch == 0:
                train_loader = self.train_loader_skipped
            else:
                train_loader = self.train_loader

            for batch in train_loader:
                # end training
                if self.data_step >= self.config["train"]["num_training_steps"]:
                    break
                # data to device
                for k, v in batch.items():
                    batch[k] = v.to(self.accelerator.device, non_blocking=True)
                self.model.train()
                # train step
                with self.accelerator.accumulate(self.model):
                    losses = self.train_step(batch)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1

                # log
                if (
                    self.data_step % self.log_interval == 0
                    and self.data_step > 0
                    and self.accelerator.is_main_process
                ):
                    self.log(losses)
                # eval/vis model output
                if (
                    self.data_step % self.eval_interval == 0
                    and self.accelerator.is_main_process
                    and self.train_and_eval
                ):
                    self.eval()
                # save state
                self.accelerator.wait_for_everyone()
                if self.data_step % self.save_interval == 0 and self.data_step > 0:
                    self.accelerator.save_state(os.path.join(self.work_dir, 'checkpoint_epoch{}'.format(self.epoch)))
                
                self.data_step += 1
            self.epoch += 1
        wandb.finish()

    def log(self, losses):
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = (
            self.config["train"]["train_batch_size"]
            * self.log_interval
            * self.config["data"]["seq_length"]
        )
        wandb.log({"Training/Token per second per gpu": tokens / cost_time})
        for k, v in losses.items():
            wandb.log({"Losses/{}".format(k): v})
        current_lr = self.optim.param_groups[0]["lr"]
        wandb.log({"Training/LR": current_lr})
        if self.optim.scaler is not None:
            wandb.log({"Training/Loss Scale": self.optim.scaler.get_scale()})
        wandb.log({"Training/Data Step": self.data_step})
        wandb.log({"Training/Global Step": self.global_step})
        wandb.log({"Training/Epoch": self.epoch})
        self.accelerator.print(
            "Epoch: {}, Global Step: {}, Data Step: {}, Loss: {}, Token per second per gpu: {}".format(
                self.epoch,
                self.global_step,
                self.data_step,
                losses["total_loss"],
                tokens / cost_time,
            )
        )

    def eval(self):
        text_table = wandb.Table(columns=["question", "pred"])
        self.model.eval()
        with torch.no_grad():
            for data in val_set_pretrain:
                raw_inputs = data
                inputs = self.tokenizer(
                    raw_inputs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                input_length = inputs["input_ids"].shape[1]
                for k, v in inputs.items():
                    inputs[k] = v.to(self.accelerator.device)
                self.accelerator.wait_for_everyone()
                pred = self.model.generate(
                    **inputs, max_new_tokens=256, do_sample=True, repetition_penalty=2.0
                )
                self.accelerator.wait_for_everyone()
                pred = pred[0, input_length:]
                pred = self.tokenizer.decode(pred.cpu(), skip_special_tokens=True)
                text_table.add_data(raw_inputs, pred)
                print(raw_inputs, '\n', pred, '\n')
        wandb.log({"Predictions on {}".format(self.global_step): text_table})

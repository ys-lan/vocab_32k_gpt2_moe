"""Fixed prompt suites used for qualitative checks during and after training.

`val_set_pretrain` holds raw completion prompts for the base model: literature, everyday reasoning, arithmetic,
science, code and a multiple-choice item, so that a single sweep exposes both fluency and knowledge regressions.

`val_set_sft` holds prompts already wrapped in the instruction template produced by `dataset.dataset.instruct_transform`
("### Instruction:\n...\n\n### System:\n"), which is what the SFT and DPO checkpoints expect at inference time.
"""

val_set_pretrain = [
    "The sun sets behind the mountains, and",
    "Once upon a time in a kingdom by the sea, there lived a young cartographer who",
    "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that",
    "A teacher is someone who passes on knowledge and resolves doubts. No one is born knowing everything, so",
    "When I woke up, I found myself in a completely unfamiliar place. There was nobody around, only a single note on the table.",
    "In recent years, there has been increasing interest in the use of machine learning to",
    "Deep learning has made remarkable progress on image recognition, yet complex scenes remain difficult, for example",
    "Given the three numbers 1, 2 and 3, their average is",
    "Tom had 15 apples. He gave two apples to each of three friends, then ate one himself. How many apples does he have left?",
    "According to Newton's second law, the acceleration of an object is equal to",
    "Carbon nanotubes are a novel material with highly unusual electronic and optical properties. Over the past few years, our understanding of carbon",
    "Below is an implementation of quicksort in Python:",
    "The quantum many-body problem is a fundamental problem in condensed matter physics. Despite decades of research, there is still no exact solution to this problem for large systems. In this paper, we propose a novel approach based on",
    "Below is an example that trains a text classification model with PyTorch and a Transformer encoder:\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset",
    "The following are multiple choice questions about accounting. Please pick the correct answer.\nWhich of the following statements about the basic principles of tax law is incorrect? ____.\nA. The principle of statutory taxation covers both statutory tax elements and the legality of tax administration\nB. The principle of tax fairness derives from the principle of equality before the law\nC. The principle of tax efficiency covers both economic efficiency and administrative efficiency\nD. Tax authorities collect taxes under statutory procedures and may freely decide to reduce, suspend or waive taxes\nAnswer: D\nWhich of the following cannot strengthen a company's core competitiveness? ____.\nA. Product differentiation\nB. Buying a production patent\nC. Innovating production technology\nD. Hiring a production outsourcing vendor\nAnswer:",
]

val_set_sft = [
    "### Instruction:\n\nWrite a 500-word essay about the value of failure.\n\n### System:\n",
    "### Instruction:\n\nMy stomach hurts. What should I do?\n\n### System:\n",
    "### Instruction:\n\nExplain the difference between supervised fine-tuning and preference optimization to someone new to machine learning.\n\n### System:\n",
    "### Instruction:\n\nHuman: What are some cuss words in english?\n\nAssistant: Here's an incomplete list.\n\nAss, dick, bugger, crap, damn, bastard, jerk, moron, idiot, git, prat, twit, tosser, wanker, scumbag, sod.\n\nHuman: What's your favorite one?\n\nAssistant: ### System:\n",
]

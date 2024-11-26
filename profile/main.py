import numpy as np
from generate_dataset import generate_dataset
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"0": 0, "1": 1}
        self.inv_vocab = {0: "0", 1: "1"}

    def encode(self, text):
        return [self.vocab[char] for char in text.split()]

    def decode(self, tokens):
        return " ".join([self.inv_vocab[token] for token in tokens])


class CoinFlipDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


dataset = generate_dataset(prob_head=0.666, sequence_length=10, num_sequences=1000)
tokenizer = SimpleTokenizer()
tokenized_dataset = [tokenizer.encode(seq) for seq in dataset]

train_dataset = CoinFlipDataset(tokenized_dataset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define GPT-2
config = GPTConfig(vocab_size=2, block_size=10, n_layer=1, n_head=1, n_embd=32)
model = GPT(config)

trainer_config = TrainerConfig(
    max_epochs=10,
    batch_size=16,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    lr_decay=True,
    warmup_tokens=512,
    final_tokens=20000,
    num_workers=2,
)
trainer = Trainer(model, train_loader, None, config=trainer_config)

# Train Model
trainer.train()

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
) as prof:
    trainer.train()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

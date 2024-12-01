import torch
from torch.utils.data import Dataset, DataLoader
import random


class GenerateDataset(Dataset):
    def __init__(self, num_samples, sequence_length, p_heads):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.p_heads = p_heads

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = torch.bernoulli(
            torch.full((self.sequence_length,), self.p_heads)
        ).long()
        return sequence


num_samples = 1000
sequence_length = 10
p_heads = 0.666

dataset = GenerateDataset(num_samples, sequence_length, p_heads)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=2,
    block_size=sequence_length,
    n_layer=1,
    n_head=2,
    n_embd=16,
)
model = GPT(config)

import lightning as L
from torch.nn import functional as F
from torch.optim import Adam, SGD


class CoinFlipModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits, loss = self.model(inputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=5e-4)


pl_model = CoinFlipModel(model)

# Training setup
trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    profiler="simple",
    enable_progress_bar=False,
)

# Train the model
trainer.fit(model=pl_model, train_dataloaders=dataloader)

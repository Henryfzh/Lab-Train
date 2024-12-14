import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lightning import Trainer
import lightning as L
from torch.nn import functional as F
from torch.optim import Adam, SGD
from model import GPT, GPTConfig
from lightning.pytorch.profilers import SimpleProfiler


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


class CoinFlipModel(L.LightningModule):
    def __init__(self, model, optimizer_type):
        super().__init__()
        self.model = model
        self.optimizer_type = optimizer_type
        self.training_times = []  # Store timing data

    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits, loss = self.model(inputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "SGD":
            return SGD(self.model.parameters(), lr=5e-4)
        elif self.optimizer_type == "Adam":
            return Adam(self.model.parameters(), lr=5e-4)

    def optimizer_zero_grad(self, *args, **kwargs):
        # Manually time the zero_grad operation
        start_time = time.time()
        super().optimizer_zero_grad(*args, **kwargs)
        elapsed_time = time.time() - start_time
        self.training_times.append(elapsed_time)  # Store the time


def main():
    def train_model(
        optimizer_type, num_samples=1000, sequence_length=10, p_heads=0.666
    ):
        dataset = GenerateDataset(num_samples, sequence_length, p_heads)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        config = GPTConfig(
            vocab_size=2,
            block_size=sequence_length,
            n_layer=1,
            n_head=2,
            n_embd=16,
        )
        model = GPT(config)
        pl_model = CoinFlipModel(model, optimizer_type)

        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            profiler=SimpleProfiler(),
            enable_progress_bar=False,
        )

        trainer.fit(model=pl_model, train_dataloaders=dataloader)
        return pl_model.training_times

    sgd_times = train_model("SGD")
    adam_times = train_model("Adam")

    print(f"sgd_times: {sgd_times}")
    print(f"adam_times: {adam_times}")

    def plot_time_distributions(sgd_times, adam_times):

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        data = {
            "SGD": sgd_times,
            "Adam": adam_times,
        }
        sns.boxplot(data=data)
        plt.title("Time Distribution for SGD vs Adam Optimizer")
        plt.ylabel("Time (s)")
        plt.xlabel("Optimizer")
        boxplot_path = os.path.join("boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()

        def bootstrap(data, n_bootstraps=1000):
            return [
                np.mean(np.random.choice(data, len(data), replace=True))
                for _ in range(n_bootstraps)
            ]

        sgd_bootstrap = bootstrap(sgd_times)
        adam_bootstrap = bootstrap(adam_times)

        plt.figure(figsize=(12, 6))
        sns.histplot(
            sgd_bootstrap, kde=True, label="SGD Mean Distribution", color="blue"
        )
        sns.histplot(
            adam_bootstrap, kde=True, label="Adam Mean Distribution", color="orange"
        )
        plt.legend()
        plt.title("Bootstrap Mean Distributions")
        plt.xlabel("Mean Time (s)")
        plt.ylabel("Frequency")
        histplot_path = os.path.join("bootstrap.png")
        plt.savefig(histplot_path)
        plt.close()

    plot_time_distributions(sgd_times, adam_times)


if __name__ == "__main__":
    main()

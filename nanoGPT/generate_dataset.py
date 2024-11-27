import numpy as np


def generate_dataset(prob_head=0.666, sequence_length=10, num_sequences=1000):
    sequences = np.random.choice(
        [0, 1], size=(num_sequences * sequence_length), p=[1 - prob_head, prob_head]
    )
    return sequences


data = generate_dataset()

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Save the data to binary files
train_data.tofile("train.bin")
val_data.tofile("val.bin")

import torch

# Configuration file for GPT model training on the coin flip dataset.

# ==================== Data Parameters ====================
out_dir = "out-coinflip"  # Directory to save checkpoints and logs
eval_interval = 20  # Evaluate the model every 500 iterations
eval_iters = 10  # Number of iterations to compute validation loss
log_interval = 10  # Log training progress every 10 iterations

# Path to the data directory
data_dir = "data/coinflip"  # Directory containing train.bin and val.bin

# Ensure data directory is valid
import os

assert os.path.exists(data_dir), f"Data directory '{data_dir}' does not exist!"
assert os.path.exists(
    os.path.join(data_dir, "train.bin")
), "Training data (train.bin) not found in data_dir!"
assert os.path.exists(
    os.path.join(data_dir, "val.bin")
), "Validation data (val.bin) not found in data_dir!"

# ==================== Training Parameters ====================
always_save_checkpoint = False  # Do not save checkpoints unnecessarily
wandb_log = False  # Disable Weights & Biases logging
wandb_project = "coinflip-gpt"
wandb_run_name = "run1"

batch_size = 8  # Reduced batch size
block_size = 32  # Reduced block size

# ==================== Model Parameters ====================
n_layer = 1  # Number of transformer layers (already minimal)
n_head = 1  # Reduced number of attention heads
n_embd = 16  # Reduced embedding dimension
dropout = 0.0  # Dropout rate
vocab_size = 2  # Vocabulary size (0 and 1 for coin flip task)

# ==================== Optimizer Parameters ====================
learning_rate = 1e-3  # Initial learning rate
max_iters = 10  # Reduced maximum number of training iterations
lr_decay_iters = 10  # Iterations to decay learning rate to min_lr
min_lr = 1e-4  # Minimum learning rate
beta1 = 0.9  # Adam optimizer beta1 parameter
beta2 = 0.95  # Adam optimizer beta2 parameter

# ==================== Miscellaneous ====================
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
compile = False  # Disable PyTorch 2.0 compile for this test

# Ensure device is valid
assert device in ["cuda", "cpu"], "device must be 'cuda' or 'cpu'!"

# ==================== Debugging and Additional Logs ====================
# Print the configuration for debugging
print(f"Configuration Loaded:")
print(f"Data Directory: {data_dir}")
print(f"Output Directory: {out_dir}")
print(f"Batch Size: {batch_size}, Block Size: {block_size}")
print(f"Model: {n_layer} layers, {n_head} heads, {n_embd} embedding size")
print(f"Training on {device.upper()} with learning rate {learning_rate}")

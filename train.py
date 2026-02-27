"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
# import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
# if config.to_log:
#     wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""
import matplotlib.pyplot as plt

# 1. Setup device and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Lists to store metrics for plotting
train_losses = []
val_losses = []
iter_list = []

iter_num = 0
model.train()

print(f"Starting training for {MODEL} on {device}...")

# 3. Training Loop
while iter_num < config.max_iter:
    for x, y in train_dataloader:
        if iter_num >= config.max_iter:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Reshape for CrossEntropyLoss (B, T, V) -> (B*T, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if config.to_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
        optimizer.step()
        
        # 4. Periodic Logging & Validation
        if iter_num % config.log_interval == 0:
            # Track training loss
            train_losses.append(loss.item())
            iter_list.append(iter_num)
            
            # Optional: Calculate Validation Loss
            model.eval()
            with torch.no_grad():
                # Get one batch from val_dataloader
                val_x, val_y = next(iter(val_dataloader))
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                vB, vT, vV = val_logits.shape
                v_loss = criterion(val_logits.view(vB * vT, vV), val_y.view(vB * vT))
                val_losses.append(v_loss.item())
            model.train()

            print(f"Iter {iter_num}: Train Loss {loss.item():.4f} | Val Loss {v_loss.item():.4f}")

        iter_num += 1

# 5. Save the final plot locally
plt.figure(figsize=(10, 5))
plt.plot(iter_list, train_losses, label='Train Loss')
plt.plot(iter_list, val_losses, label='Val Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'{MODEL} Training and Validation Loss')
plt.legend()
plt.savefig(config.save_path / 'loss_curve.png')
plt.show()

print(f"Training finished. Plot saved to {config.save_path}/loss_curve.png")
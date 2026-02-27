"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
# import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "minigpt"  # bigram or minigpt

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# Using a fallback if learning_rate is missing from config
lr = getattr(config, 'learning_rate', 3e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

train_losses = []
val_losses = []
steps = []

iter_num = 0
model.train()

print(f"Starting training on {device}...")

while iter_num < config.max_iter:
    for x, y in train_dataloader:
        if iter_num >= config.max_iter: break
        
        x, y = x.to(device), y.to(device)
        logits = model(x)
        
        # Reshape for CrossEntropy
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Log and Validate
        if iter_num % config.log_interval == 0:
            steps.append(iter_num)
            train_losses.append(loss.item())
            
            # Simple Val Loss check
            model.eval()
            with torch.no_grad():
                vx, vy = next(iter(eval_dataloader))
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                v_loss = criterion(v_logits.view(-1, v_logits.size(-1)), vy.view(-1))
                val_losses.append(v_loss.item())
            model.train()
            
            print(f"Iter {iter_num}: Train Loss {loss.item():.4f} | Val Loss {v_loss.item():.4f}")
            
        iter_num += 1

# Save data for the Notebook to use
history = {"steps": steps, "train": train_losses, "val": val_losses}
with open("loss_history.json", "w") as f:
    json.dump(history, f)

print("Training complete. Data saved to loss_history.json")
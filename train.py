"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

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
if config.to_log:
    wandb.init(project="dl2_proj3")


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

# Move model to device (GPU or CPU)
model.to(device)

# 1. Define Loss Function and Optimizer
# CrossEntropyLoss expects (Batch, Vocab_Size, Seq_Len) for multidimensional inputs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 3e-4)

# Training loop variables
iter_num = 0
best_train_loss = float('inf')

model.train()

# Standard training loop over iterations
while iter_num < config.max_iter:
    for batch_idx, (x, y) in enumerate(train_dataloader):
        if iter_num >= config.max_iter:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Reshape logits and targets for CrossEntropyLoss
        # Logits: (B, T, V) -> (B*T, V)
        # Targets: (B, T) -> (B*T)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Optional: Gradient clipping if set in config
        if config.to_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
        optimizer.step()
        
        # 2. Logging
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")
            if config.to_log:
                wandb.log({"train_loss": loss.item(), "iter": iter_num})
                
        # 3. Save Model with Best Training Loss
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            if iter_num > 0: # Avoid saving at the very first iteration
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_train_loss,
                    'iter': iter_num,
                }
                torch.save(checkpoint, config.save_path / "best_model.pt")

        # Optional: Save every config.save_iterations
        if iter_num % config.save_iterations == 0 and iter_num > 0:
            torch.save({'model_state_dict': model.state_dict()}, config.save_path / f"model_iter_{iter_num}.pt")

        iter_num += 1

print(f"Training completed. Best loss: {best_train_loss:.4f}")

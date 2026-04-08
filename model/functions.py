# I want to define some useful functions for the model

import pandas as pd
import torch as torch
from torch.nn import functional as F
from tqdm import tqdm

def pinball_loss(y_hat:torch.Tensor, y:torch.Tensor, h_cost:float, l_cost:float) -> torch.Tensor:
    """
    This will calculate the pinball loss i.e.
    if y_hat > y, then the loss is h_cost * (y_hat - y)
    if y_hat < y, then the loss is l_cost * (y - y_hat)
    """
    return torch.mean(h_cost * (y_hat - y).clamp(min=0) + l_cost * (y - y_hat).clamp(min=0))

def pinball_loss_sum(y_hat:torch.Tensor, y:torch.Tensor, h_cost:float, l_cost:float) -> torch.Tensor:
    """
    This will calculate the pinball loss i.e.
    if y_hat > y, then the loss is h_cost * (y_hat - y)
    if y_hat < y, then the loss is l_cost * (y - y_hat)
    """
    return torch.sum(h_cost * (y_hat - y).clamp(min=0) + l_cost * (y - y_hat).clamp(min=0))

def pinball_loss_tensor(y_hat:torch.Tensor, y:torch.Tensor, h_cost:float, l_cost:float) -> torch.Tensor:
    """
    This will calculate the pinball loss i.e.
    if y_hat > y, then the loss is h_cost * (y_hat - y)
    if y_hat < y, then the loss is l_cost * (y - y_hat)
    """
    return h_cost * (y_hat - y).clamp(min=0) + l_cost * (y - y_hat).clamp(min=0)

def rmse(y_hat:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """
    This will calculate the root mean squared error
    """
    return torch.sqrt(torch.mean((y_hat - y) ** 2))

def train(
    net:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    loss:callable,
    train_loader:torch.utils.data.DataLoader,
    val_loader:torch.utils.data.DataLoader,
    epochs:int,
    eval_interval:int,
    device:str,
    use_tqdm: bool = True,
    use_one_cycle_lr: bool = False,
    one_cycle_max_lr: float | None = None,
) -> tuple[list[float], list[float]]:
    """
    This will train the model.

    If use_one_cycle_lr is True, applies OneCycleLR with the same hyperparameters as
    OneCycleLR_pct0.2_div10 (pct_start=0.2, cosine, div_factor=10, final_div_factor=1e4),
    stepped once per training step. max_lr defaults to the optimizer's initial LR.
    """
    
    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []

    one_cycle_scheduler = None
    if use_one_cycle_lr:
        max_lr = (
            one_cycle_max_lr
            if one_cycle_max_lr is not None
            else optimizer.param_groups[0]["lr"]
        )
        one_cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=epochs,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1e4,
        )

    # Initialize iterator for the train loader
    train_loader_iter = iter(train_loader)

    pbar = tqdm(range(epochs), desc="Training", unit="step") if use_tqdm else range(epochs)
    for step in pbar:
        # Get the next batch of data
        try:
            x, y = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            x, y = next(train_loader_iter)

        # Move data to device
        x, y = x.to(device), y.to(device)

        # Perform a forward pass and compute the loss
        optimizer.zero_grad()
        y_hat = net(x)
        train_loss = loss(y_hat, y)
        train_loss.backward()
        optimizer.step()
        if one_cycle_scheduler is not None:
            one_cycle_scheduler.step()

        # Store the loss
        train_losses.append(train_loss.item())
        if use_tqdm:
            pbar.set_postfix(train_loss=f"{train_loss.item():.4f}")

        # Evaluate the model on the validation set
        if eval_interval is not None:
            if step % eval_interval == 0:
                net.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        y_hat = net(x)
                        val_loss_sum += loss(y_hat, y).item()
                        val_batches += 1
                net.train()
                mean_val_loss = val_loss_sum / val_batches if val_batches else 0.0
                val_losses.append(mean_val_loss)
                if use_tqdm:
                    pbar.set_postfix(train_loss=f"{train_loss.item():.4f}", val_loss=f"{mean_val_loss:.4f}")

    return train_losses, val_losses

def get_test_loss(net:torch.nn.Module, test_loader:torch.utils.data.DataLoader, loss:callable, device:str) -> list[float]:
    """
    Get the loss for the test set and return a list of losses
    """
    test_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            test_loss = loss(y_hat, y)
            test_losses.append(test_loss.item())
    return test_losses


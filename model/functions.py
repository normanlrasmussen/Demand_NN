# I want to define some useful functions for the model

import pandas as pd
import torch as torch
from torch.nn import functional as F

def pinball_loss(y_hat:torch.Tensor, y:torch.Tensor, h_cost:float, l_cost:float) -> torch.Tensor:
    """
    This will calculate the pinball loss i.e.
    if y_hat > y, then the loss is h_cost * (y_hat - y)
    if y_hat < y, then the loss is l_cost * (y - y_hat)
    """
    return torch.mean(h_cost * (y_hat - y).clamp(min=0) + l_cost * (y - y_hat).clamp(min=0))

def rmse(y_hat:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """
    This will calculate the root mean squared error
    """
    return torch.sqrt(torch.mean((y_hat - y) ** 2))

def train(
    net:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    loss:callable,
    train_loader:torch.Dataloader,
    val_loader:torch.Dataloader,
    epochs:int,
    eval_interval:int,
    device:str
) -> None:
    """
    This will train the model
    """
    
    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []

    # Initialize iterator for the train loader
    train_loader_iter = iter(train_loader)

    for step in range(epochs):
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

        # Store the loss
        train_losses.append(train_loss.item())

        # Evaluate the model on the validation set
        if step % eval_interval == 0:
            # Init the val loader iterator
            val_loader_iter = iter(val_loader)

            with torch.no_grad():
                for x, y in val_loader_iter:
                    x, y = x.to(device), y.to(device)
                    y_hat = net(x)
                    val_loss = loss(y_hat, y)
                    val_losses.append(val_loss.item())

    return train_losses, val_losses
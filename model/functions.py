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
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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
        x, y = x.to(device), y.to(device).long()

        # Perform a forward pass and compute the loss
        optimizer.zero_grad() 
        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        # Store the loss
        train_losses.append(loss.item())

        # Evaluate the model on the validation set
        if step % eval_interval == 0:
            val_loss, val_acc = val_net(net, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

    return train_losses, train_accuracies, val_losses, val_accuracies
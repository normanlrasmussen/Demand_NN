# This will split the data into train, val, test and put it all in data loaders

import pandas as pd
import torch
from torch._C import parse_schema
from torch.utils.data import DataLoader, Dataset
try:
    from .data_creation import create_data
except ImportError:
    from data_creation import create_data
from typing import Tuple

class DemandDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Extract numeric features and target as writable float32 arrays
        x_np = df.drop(columns=["date", "sales"]).to_numpy(dtype="float32", copy=True)
        y_np = df["sales"].to_numpy(dtype="float32", copy=True)

        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_dataloader(
    input_file:str="../data/demand_data.csv", 
    specs: Tuple[str, ...]=(), 
    date_splits: Tuple[str, str] = ("2017-01-01", "2017-06-01"),
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
    ):
    """
    Create the dataloaders for the train, val, and test sets
    """

    # Create the data
    df = create_data(input_file=input_file, specs=specs)
    train_df = df[df['date'] < date_splits[0]]
    val_df = df[(df['date'] >= date_splits[0]) & (df['date'] < date_splits[1])]
    test_df = df[df['date'] >= date_splits[1]]

    # Create the datasets
    train_dataset = DemandDataset(train_df)
    val_dataset = DemandDataset(val_df)
    test_dataset = DemandDataset(test_df)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    return train_loader, val_loader, test_loader

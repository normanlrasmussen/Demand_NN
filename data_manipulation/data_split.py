# This will split the data into train, val, test and put it all in data loaders

import numpy as np
import pandas as pd
import torch
from torch._C import parse_schema
from torch.utils.data import DataLoader, Dataset
try:
    from .data_creation import create_data_all_data, create_data_consolidated_by_item, create_data_consolidated_by_store, create_data_consolidated_by_both
except ImportError:
    from data_creation import create_data_all_data, create_data_consolidated_by_item, create_data_consolidated_by_store, create_data_consolidated_by_both
from typing import Tuple

class DemandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, drop_columns: list[str], target_columns: list[str]):

        # Extract numeric features and target as writable float32 arrays
        x_np = df.drop(columns=drop_columns).to_numpy(dtype="float32", copy=True)
        y_np = df[target_columns].to_numpy(dtype="float32", copy=True)

        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np)

        # Save the column names for x, y
        self.x_columns = df.drop(columns=drop_columns).columns
        self.y_columns = target_columns

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_dataloader(
    input_file:str="../data/demand_data.csv", 
    specs: Tuple[str, ...]=(), # specs for the data creation
    date_splits: Tuple[str, str] = ("2017-01-01", "2017-06-01"),
    batch_size: int = 4,
    test_batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False, 
    data_mask: list[tuple[str, int]] | None = None,  # list of boolean masks for filtering
    combine_items: bool = False, # Put all items into one row
    combine_stores: bool = False, # Put all stores into one row
    ):
    """
    Create the dataloaders for the train, val, and test sets
    """

    # Get the data
    if not combine_items and not combine_stores:
        df, drop_columns, target_columns = create_data_all_data(input_file=input_file, specs=specs, data_mask=data_mask)
    elif combine_items and not combine_stores:
        df, drop_columns, target_columns = create_data_consolidated_by_item(input_file=input_file, specs=specs, data_mask=data_mask)
    elif not combine_items and combine_stores:
        df, drop_columns, target_columns = create_data_consolidated_by_store(input_file=input_file, specs=specs, data_mask=data_mask)
    elif combine_items and combine_stores:
        df, drop_columns, target_columns = create_data_consolidated_by_both(input_file=input_file, specs=specs, data_mask=data_mask)

    # Split the data into train, val, and test
    train_df = df[df['date'] < date_splits[0]]
    val_df = df[(df['date'] >= date_splits[0]) & (df['date'] < date_splits[1])]
    test_df = df[df['date'] >= date_splits[1]]

    # Create the datasets
    train_dataset = DemandDataset(train_df, drop_columns=drop_columns, target_columns=target_columns)
    val_dataset = DemandDataset(val_df, drop_columns=drop_columns, target_columns=target_columns)
    test_dataset = DemandDataset(test_df, drop_columns=drop_columns, target_columns=target_columns)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloader(batch_size=8, combine_items=True, combine_stores=True)

# This will split the data into train, val, test and put it all in data loaders

import numpy as np
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
    def __init__(self, df: pd.DataFrame, combine_items: bool = False, combine_stores: bool = False):

        # No combining – standard single-output target (sales)
        if not combine_items and not combine_stores:
            drop_columns = ["date", "sales", "store", "item"]
            target_columns = ["sales"]

        # Combine all items into a multi-output target per (date, store)
        elif combine_items and not combine_stores:
            target_columns = [f"item_{c}" for c in df["item"].unique()]
            wide = df.pivot(
                index=["date", "store"],
                columns="item",
                values="sales",
            )
            wide.columns = [f"item_{c}" for c in wide.columns]
            df = wide.reset_index()
            print(df.head())
            # Keep non-target columns (e.g., store) as features
            drop_columns = ["date"] + target_columns

        # Combine all stores into a multi-output target per (date, item)
        elif combine_stores and not combine_items:
            target_columns = [f"store_{c}" for c in df["store"].unique()]
            wide = df.pivot(
                index=["date", "item"],
                columns="store",
                values="sales",
            )
            wide.columns = [f"store_{c}" for c in wide.columns]
            df = wide.reset_index()
            # Keep non-target columns (e.g., item) as features
            drop_columns = ["date"] + target_columns

        # Combine both stores and items into one wide vector per date:
        # columns are store_{store}_item_{item}
        elif combine_items and combine_stores:
            wide = df.pivot(
                index=["date"],
                columns=["store", "item"],
                values="sales",
            )
            wide.columns = [f"store_{s}_item_{i}" for (s, i) in wide.columns]
            df = wide.reset_index()
            target_columns = [c for c in df.columns if c.startswith("store_")]
            drop_columns = ["date"] + target_columns




        # Extract numeric features and target as writable float32 arrays
        x_np = df.drop(columns=drop_columns).to_numpy(dtype="float32", copy=True)
        y_np = df[target_columns].to_numpy(dtype="float32", copy=True)

        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np)

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
    data_mask: list[pd.Series] | None = None,  # list of boolean masks for filtering
    combine_items: bool = False, # Put all items into one row
    combine_stores: bool = False, # Put all stores into one row
    ):
    """
    Create the dataloaders for the train, val, and test sets
    """

    # Create the data
    df = create_data(input_file=input_file, specs=specs)

    # Apply the data mask
    if data_mask is not None:
        combined_mask = np.logical_and.reduce(data_mask)
        df = df[combined_mask]

    # Split the data into train, val, and test
    train_df = df[df['date'] < date_splits[0]]
    val_df = df[(df['date'] >= date_splits[0]) & (df['date'] < date_splits[1])]
    test_df = df[df['date'] >= date_splits[1]]

    # Create the datasets
    train_dataset = DemandDataset(train_df, combine_items=combine_items, combine_stores=combine_stores)
    val_dataset = DemandDataset(val_df, combine_items=combine_items, combine_stores=combine_stores)
    test_dataset = DemandDataset(test_df, combine_items=combine_items, combine_stores=combine_stores)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloader(batch_size=8, combine_items=True, combine_stores=True)

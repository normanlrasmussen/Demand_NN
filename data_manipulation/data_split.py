# This will split the data into train, val, test and put it all in data loaders

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_creation import create_data

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

# TODO create function to call create data, and make dataloader with desired specs



if __name__ == "__main__":
    df = create_data()
    dataset = DemandDataset(df)
    print(dataset[0])
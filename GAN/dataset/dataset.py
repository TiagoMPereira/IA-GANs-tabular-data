import torch
from torch.utils.data import Dataset

from pandas import DataFrame

# Define the dataset
class TabularDataset(Dataset):
    def __init__(self, data: DataFrame):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
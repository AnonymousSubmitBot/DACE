import numpy as np
from torch.utils.data import Dataset


class ZeroOneProblemData(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, split: str):
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        x = x * 2 - 1
        self.x = x[:int(len(x) * 0.75)] if split == "train" else (x[int(len(x) * 0.75):] if split == "valid" else x)
        self.y = y[:int(len(y) * 0.75)] if split == "train" else (y[int(len(y) * 0.75):] if split == "valid" else y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y

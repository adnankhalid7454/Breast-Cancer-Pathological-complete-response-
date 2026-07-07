"""
Shared utilities: reproducibility, PyTorch Dataset, and DataLoader collate function.
"""
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int = 42):
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id, base_seed=42):
    """DataLoader worker seeding hook (used via functools.partial or lambda)."""
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


class TabularDataset(Dataset):
    """
    Wraps a list of per-modality feature arrays + a label array.
    Each item returns [modality_1_row, modality_2_row, ...], label.
    """
    def __init__(self, features_list, y):
        self.features_list = [torch.tensor(f, dtype=torch.float32) for f in features_list]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [f[idx] for f in self.features_list], self.y[idx]


def collate_fn(batch):
    """Custom collate: stacks each modality separately across the batch."""
    features = list(zip(*[item[0] for item in batch]))
    labels = torch.stack([item[1] for item in batch])
    return [torch.stack(m) for m in features], labels

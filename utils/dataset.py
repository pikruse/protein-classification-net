import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class HalideDataset(Dataset):
    """
    Input is a protein sequence with a binding site region mask; Output is a tuple of (occupancy, halide)
    """

    def __init__(self, pdb_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
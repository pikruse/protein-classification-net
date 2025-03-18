import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from DGXutils import GetFileNames

import codebase.projects.proteins.ProteinClassificationNet.utils.model as model
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import (
    ESMProtein,
    GenerationConfig,
)

# TO-DO: MAKE SURE THE ESM FUNCTIONS ARE CORRECT

class HalideDataset(Dataset):
    """
    Input is a protein sequence with a binding site region mask; Output is a tuple of (occupancy, halide)
    """

    def __init__(self, pdb_dir):
        self.pdb_dir = pdb_dir
        self.pdb_files = GetFileNames(pdb_dir, '.pdb.gz')

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        # get pdb idx
        pdb_file = self.pdb_files[idx]

        # binding site mask
        
        # form into dict
        esm_dict = {
            "sequence": ProteinChain(pdb_file).sequence, # replace with the right code for getting sequence
            "structure": ProteinChain(pdb_file).structure, # `...` structure
            "function": ProteinChain(pdb_file).function, # `...` function
            "halide": ProteinChain(pdb_file).halide, # `...` halide
        }

        return esm_dict 
    
class QuantumHalideDataset(Dataset):
    # for tyler's quantum model
    pass


# example code

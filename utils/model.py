# load necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# ----------------  
## MHT TRANSFORMER
# ----------------
class Block(nn.Module):
    """
    FC block with linear + bn + dropout + activation
    """
    def __init__(self,)

class MHTTransformer(nn.Module):

    def __init__(self,
                 n_layers,
                 hidden_dim,
                 out_dim,
                 n_heads,):
        super(MHTTransformer, self).__init__()

        # get hyperparameters
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads

        # make layers

    # fwd pass
    def forward(self, x):
        pass

# --------------
## ESM FINE-TUNE
# --------------
class RegressionHead(nn.Module):
    def __init__(self,
                 n_layers: int = 1,
                 hidden_dim: int = 512,
                 ):
        # figure out what the output of ESM3 is (size, shape, etc.)
        # define the layers of the regression head
        pass

    def forward(self, x):
        pass

class ESMClassifier(nn.Module):

    def __init__(self,
                 model_path: str = "esm3_sm_open_v1",
                 ):
        
        self.esm = ESM3.from_pretrained(model_path)
        self.reg_head = RegressionHead()

    def forward(self, x):
        # x is our .pdb input
        seq = x["sequence"]
        structure = x["structure"]
        function = x["function"]

        # get output from esm3
        esm_out = self.esm(seq, structure, function)

        # send it through our custom classifier/regression head
        reg_out = self.reg_head(esm_out)

        return reg_out
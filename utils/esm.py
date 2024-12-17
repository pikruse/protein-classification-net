# load necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

class ESMClassifier(nn.Module):

    def __init__(self,
                 model_path: str = "esm3_sm_open_v1",
                 n_out: int = 1,
                 activation = nn.ReLU,
                 ):
        """
        Parameters:
            model_path (str): path to esm model on huggingface
            n_out (int): number of output neurons
        """
        # self.model_path = model_path
        # self.model = ESM3.from_pretrained(model_path)
        # self.tokenizer = ESM
        # self.activation = activation
        # self.output_layer = nn.Linear( ,n_out)

        ### output layer will have two neurons, one for occupancy ([0, 1]) and one for the halide ([B, C])
        pass

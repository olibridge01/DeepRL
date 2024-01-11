import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, 
                 input_dims: int, 
                 output_dims: int, 
                 hidden_dims: list = [32, 32], 
                 activation: nn.Module = nn.ReLU(), 
                 activation_last_layer: bool = False
        ):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.activation = activation

        network = []
        dims = [input_dims] + hidden_dims + [output_dims]
        for i in range(len(dims) - 1):
            network.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or activation_last_layer:
                network.append(self.activation)

        self.network = nn.Sequential(*network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)    
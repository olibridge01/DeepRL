import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN_CNN(nn.Module):
    def __init__(self,
                 input_shape: tuple = (4, 84, 84),
                 output_dims: int = 5,
                 kernel_sizes: list = [8, 4],
                 strides: list = [4, 2],
                 channels: list = [16, 32],
                 hidden_dims: list = [256],
                 activation: nn.Module = nn.ReLU(),
                 activation_last_layer: bool = False
        ):
        super(DQN_CNN, self).__init__()
        self.input_channels = input_shape[0]
        self.input_shape = input_shape[1:]
        self.output_dims = output_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.hidden_dims = hidden_dims
        self.activation = activation

        assert len(kernel_sizes) == len(strides) == len(channels), "CNN settings not consistent."

        self.layers = []
        for i in range(len(channels)):
            if i == 0:
                self.layers.append(nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i]
                ))
            else:
                self.layers.append(nn.Conv2d(
                    in_channels=channels[i-1],
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i]
                ))
            self.layers.append(self.activation)

        # Calculate feature size
        test_tensor = torch.zeros(1, *input_shape)
        for layer in self.layers:
            test_tensor = layer(test_tensor)
        self.feature_size = lambda: int(np.prod(test_tensor.size()))

        # Add fully connected layers
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(self.feature_size(), hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.layers.append(nn.Linear(hidden_dims[-1], output_dims))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


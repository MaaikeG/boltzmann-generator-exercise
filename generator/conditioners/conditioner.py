import torch.nn.functional
from torch import nn


class Conditioner(nn.Module):
    """Avery simple densely connected network."""
    def __init__(self, dim_in, dims_out, activation=nn.ReLU):
        super().__init__()

        layers = []

        for dim_out in dims_out:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(activation())

            dim_in = dim_out

        nn.ModuleList()

        self.layers = nn.Sequential(*layers)


    def forward(self, samples):
        return self.layers(samples)

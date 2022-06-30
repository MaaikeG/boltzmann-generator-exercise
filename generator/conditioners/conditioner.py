import torch.nn.functional
from torch import nn


class Conditioner(nn.Module):
    """Avery simple densely connected network."""
    def __init__(self, dims, activation=nn.ReLU):
        super().__init__()

        layers = []

        dim_in = dims[0]

        for dim_out in dims[1:]:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(activation())

            dim_in = dim_out

        nn.ModuleList()

        self.layers = nn.Sequential(*layers)


    def forward(self, samples):
        return self.layers(samples)

import torch
from .tranformer_base import Transformer


def log_jacobian_determinant(cond):
    return cond[:, 0].sum()


class AffineTransformer(Transformer):
    """Affine transformation of one half of the input, conditioned on the other
    half.

    Performs transformation of form y = A x_1 + b where A, b are defined by
    c(x_2), where c is a conditioning function on x with a two-dimensional
    output, and is modeled by a neural network.

    Parameters
    ----------
    conditioner : torch.nn.Model
        A model with output dimension 2
    """
    def __init__(self, conditioner):
        super().__init__(conditioner)


    def forward(self, pz_1, pz_2):
        cond = self.conditioner(pz_2)
        px_1 = torch.exp(cond[:, 0]) * pz_1 + cond[:, 1]
        return torch.cat(px_1, pz_2), log_jacobian_determinant(cond)


    def inverse(self, px_1, px_2):
        cond = self.conditioner(px_2)
        pz_1 = torch.exp(-cond[:, 0]) * (px_1 - cond[:, 1])
        return torch.cat(pz_1, px_2), 1 / log_jacobian_determinant(cond)

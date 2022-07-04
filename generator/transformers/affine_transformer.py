import torch
from .tranformer_base import Transformer


def log_jacobian_determinant(cond):
    return cond[:, 0].sum()


class AffineTransformer(Transformer):
    """Affine transformation of one half of the input, conditioned on the other
    half.

    Performs transformation of form y = A x_1 + b where A, b are defined by
    c_1(x_2) and c_2(x_2), where c_1 and c_2 are conditioning functions on x
    which are modeled by a neural network.

    Parameters
    ----------
    scale_conditioner : torch.nn.Model
        A model with output dimension D that scales the input values

    shift_conditioner : torch.nn.Model
        A model with output dimension D that shifts the input values

    """
    def __init__(self, shift_conditioner=None, scale_conditioner=None):
        super().__init__()
        self.shift_conditioner = shift_conditioner
        self.scale_conditioner = scale_conditioner

    def _get_scale_and_shift(self, coords):
        if self.shift_conditioner is not None:
            shift = self.shift_conditioner(coords)
        else:
            shift = torch.zeros_like(coords)

        if self.scale_conditioner is not None:
            scale = self.scale_conditioner(coords)
        else:
            scale = torch.zeros_like(coords)
        return scale, shift

    def forward(self, pz_1, pz_2):
        scale, shift = self._get_scale_and_shift(pz_2)
        px_1 = torch.exp(scale) * pz_1 + shift
        log_jac_det = scale.mean(-1)
        return px_1, log_jac_det

    def inverse(self, px_1, px_2):
        scale, shift = self._get_scale_and_shift(px_2)
        pz_1 = torch.exp(-scale) * (px_1 - shift)
        log_jac_det = -scale.mean(-1)
        return pz_1, log_jac_det

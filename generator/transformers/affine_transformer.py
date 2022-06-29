import torch
from .tranformer_base import Transformer


def log_jacobian_determinant(cond):
    return cond[:, 0].sum()


class AffineTransformer(Transformer):
    def transform(self, pz_1, pz_2, cond):
        px_1 = torch.exp(cond[:, 0]) * pz_1 + cond[:, 1]
        px = torch.cat(px_1, pz_2)
        return px, log_jacobian_determinant(cond)

    def inverse_transform(self, px_1, px_2, cond):
        pz_1 = torch.exp(-cond[:, 0]) * (px_1 - cond[:, 1])
        pz = torch.cat(pz_1, px_2)
        return pz, 1 / log_jacobian_determinant(cond)

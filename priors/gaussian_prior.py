import torch
from priors.prior_base import Prior


class GaussianPrior(Prior):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, coords):
        return 0.5 * torch.linalg.norm(coords, dim=-1) ** 2

    def sample(self, n):
        return torch.normal(mean=0, std=1, size=[n, self.dim])

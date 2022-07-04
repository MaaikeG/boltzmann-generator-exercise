import torch
from .potential_base import Potential
from util.multivariate_gaussian import MultiVariateGaussian


class NDGaussianWell(Potential):
    def __init__(self, mu_s, sigma_s):
        super().__init__()

        self.n_dimensions = mu_s[0].shape[0]
        self.gaussians = []
        for mu, sigma in zip(mu_s, sigma_s):
            if mu.shape[0] != self.n_dimensions:
                raise ValueError
            if not (torch.Tensor([sigma.shape]) == self.n_dimensions).all():
                raise ValueError

            self.gaussians.append(MultiVariateGaussian(mu, sigma))

    def __call__(self, coords, **kwargs):
        return sum([self._cutoff(-torch.log(N(coords))) for N in self.gaussians])


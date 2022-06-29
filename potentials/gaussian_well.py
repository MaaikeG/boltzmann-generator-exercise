import torch
from .potential_base import Potential


class MultiVariateGaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        n = mu.shape[0]
        self.sigma_inv = torch.linalg.inv(sigma)
        self.N = torch.sqrt((2*torch.pi)**n * torch.linalg.det(sigma))

    def __call__(self, coords):
        fac = torch.einsum('...k,kl,...l->...', coords - self.mu, self.sigma_inv, coords - self.mu)
        return torch.exp(-fac / 2) / self.N


class NDGaussianWell(Potential):
    def __init__(self, mu_s, sigma_s):
        self.n_dimensions = mu_s[0].shape[0]
        self.potentials = []
        for mu, sigma in zip(mu_s, sigma_s):
            if mu.shape[0] != self.n_dimensions:
                raise ValueError
            if not (torch.Tensor([sigma.shape]) == self.n_dimensions).all():
                raise ValueError

            self.potentials.append(MultiVariateGaussian(mu, sigma))

    def potential(self, coords, **kwargs):
        return sum([U(coords) for U in self.potentials])


class TwoDimensionalDoubleWell(NDGaussianWell):
    def __init__(self):
        mu_s = [torch.Tensor([0., 0.]), torch.Tensor([2.5, 2.5])]
        sigma_s = [torch.Tensor([[1., 0.], [0.,  1.]]), torch.Tensor([[1., 0.], [0.,  1.]])]
        super().__init__(mu_s, sigma_s)

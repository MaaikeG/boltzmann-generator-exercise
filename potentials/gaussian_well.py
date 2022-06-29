import numpy as np
from .potential_base import Potential


class MultiVariateGaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        n = mu.shape[0]
        self.sigma_inv = np.linalg.inv(sigma)
        self.N = np.sqrt((2*np.pi)**n * np.linalg.det(sigma))

    def __call__(self, coords):
        fac = np.einsum('...k,kl,...l->...', coords - self.mu, self.sigma_inv, coords - self.mu)
        return np.exp(-fac / 2) / self.N


class NDGaussianWell(Potential):
    def __init__(self, mu_s, sigma_s):
        self.n_dimensions = mu_s[0].shape
        self.potentials = []
        for mu, sigma in zip(mu_s, sigma_s):
            if mu.shape != self.n_dimensions:
                raise ValueError
            if not (np.asarray(sigma.shape) == self.n_dimensions).all():
                raise ValueError

            self.potentials.append(MultiVariateGaussian(mu, sigma))

    def potential(self,  *args, **kwargs):
        return sum([U(args) for U in self.potentials])


class TwoDimensionalDoubleWell(NDGaussianWell):
    def __init__(self):
        mu_s = [np.asarray([0., 0.]), np.asarray([2.5, 2.5])]
        sigma_s = [np.asarray([[1., 0.], [0.,  1.]]), np.asarray([[1., 0.], [0.,  1.]])]
        super().__init__(mu_s, sigma_s)

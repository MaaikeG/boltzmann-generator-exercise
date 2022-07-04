import torch


class MultiVariateGaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        n = mu.shape[0]
        self.sigma_inv = torch.linalg.inv(sigma)
        self.N = torch.sqrt((2*torch.pi)**n * torch.linalg.det(sigma))

    def __call__(self, coords):
        fac = torch.einsum('...k,kl,...l->...', coords - self.mu, self.sigma_inv, coords - self.mu)
        return torch.exp(-fac / 2) / self.N

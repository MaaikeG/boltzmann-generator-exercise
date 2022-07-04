import torch
from potentials.prior_base import Prior
from potentials.gaussian_well import NDGaussianWell
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianPrior(NDGaussianWell, Prior):

    def __init__(self, dim=1):
        self.dim = dim
        super(GaussianPrior, self).__init__(mu=torch.zeros(dim), sigma=torch.eye(dim))


    def potential(self, r):
        return 0.5 * torch.linalg.norm(r, dim=-1) ** 2


    def sample(self, n):
        return torch.normal(mean=0, std=1, size=[n, self.dim])

    # def probability(self, r):
    #     return torch.exp(self.distribution.log_prob(r))

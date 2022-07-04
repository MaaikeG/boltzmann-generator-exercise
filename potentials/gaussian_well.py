import torch
from .distribution_base import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal


class NDGaussianWell(Distribution):

    def __init__(self, mu, sigma):
        super().__init__()
        self.distribution = MultivariateNormal(loc=mu, covariance_matrix=sigma)


    def potential(self, r):
        return self._cutoff(-torch.log(self.distribution.log_prob(r)))

    # def probability(self, r):
    #     return torch.exp(self.distribution.log_prob(r))

    def log_prob(self, r):
        return self.distribution.log_prob(r)

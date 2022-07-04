import abc

import torch
from .distribution_base import Distribution


class Prior(Distribution):
    """A distribution with additional property that you can sample from it."""
    def __init__(self, **kwargs):
        super().__init__()


    @abc.abstractmethod
    def sample(self, n: int, *args) -> torch.Tensor:
        """Sample from the prior distribution."""
        pass

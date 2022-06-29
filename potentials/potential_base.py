import abc
import numpy as np


class Potential:
    def __init__(self):
        pass

    @abc.abstractmethod
    def potential(self, *args, **kwargs):
        pass

    def probability(self, *args, **kwargs):
        return np.exp(-self.potential(args, kwargs))

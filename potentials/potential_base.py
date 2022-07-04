import abc
import torch


class Potential:

    def __init__(self, cutoff_energy=1e10):
        self._cutoff_energy = cutoff_energy
        pass


    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


    def _cutoff(self, potentials):
        return torch.clamp(potentials, min=-self._cutoff_energy, max=self._cutoff_energy)


    def probability(self, *args, **kwargs):
        return torch.exp(-self.potential(args, kwargs))

import abc
import torch


class Distribution:

    def __init__(self, beta=1, cutoff_energy=1e10):
        self._beta = beta
        self._cutoff_energy = cutoff_energy


    @abc.abstractmethod
    def potential(self, r: torch.Tensor) -> torch.Tensor:
        """Return the value of the potential at the coordinates."""
        pass


    @abc.abstractmethod
    def log_prob(self, r):
        """Return the logarithm of the probability (unnormalized) at the
        coordinates. """
        return - self._beta * self.potential(r)


    def _cutoff(self, potentials):
        """Clamp the potentials (which can get very large) to cutoff maximum and
        minimum values. """
        return torch.clamp(potentials, min=-self._cutoff_energy, max=self._cutoff_energy)

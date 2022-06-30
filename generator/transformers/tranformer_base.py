import abc
import numpy as np
import torch.nn


class Transformer(torch.nn.Module):
    """Transforms one half of the input, conditioned on the other half.
    """
    def __init__(self, *args):
        super().__init__()

    @abc.abstractmethod
    def forward(self, pz_1: np.ndarray, pz_2: np.ndarray) -> (np.ndarray, float):
        """Transform the prior distribution to the target distribution.

        Parameters
        ----------
        pz_1: np.ndarray (N, D)
            The probabilities of N samples in D dimensions.
        pz_2: np.ndarray (N, k)
            The k conditioner values for the N samples.

        Returns
        -------
        px : np.ndarray(N, D)
            the forward transformed probabilities
        Log of the Jacobian determinant : float
            The natural logarithm of the determinant of the Jacobian of the
            transformation.
        """
        pass

    @abc.abstractmethod
    def inverse(self, px: np.ndarray, cond: np.ndarray) -> (np.ndarray, float):
        """Transform (normalize) samples from the target distribution back to
         the prior distribution (usually a multivariate Gaussian).

        Parameters
        ----------
        px: np.ndarray (N, D)
            The probabilities of the N samples in D dimensions.

        cond: np.ndarray (N, k)
            The k conditioner values for the N samples.

        Returns
        -------
        pz: np.ndarray(N, D)
            the inverse transformed probabilities
        Log of the Jacobian determinant : float
            The natural logarithm of the determinant of the Jacobian of the
            transformation.
        """
        pass

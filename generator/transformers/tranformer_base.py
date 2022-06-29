import abc
import numpy as np


class Transformer:
    """ Or: coupling layer """

    @abc.abstractmethod
    def transform(self, pz_1: np.ndarray, pz_2: np.ndarray, cond: np.ndarray) -> (np.ndarray, float):
        """Transform the prior distribution to the target distribution.

        Parameters
        ----------
        pz_1: np.ndarray (N, D)
            The probabilities of the first part of the N samples in D dimensions.
        pz_2: np.ndarray (M, D)
            The probabilities of the second part of the N samples in D dimensions.
        cond: np.ndarray (N, k)
            The k conditioner values for the first N samples.

        Returns
        -------
        px : np.ndarray(N+M, D)
            the transformed probabilities
        Log of the Jacobian determinant : float
            The natural logarithm of the determinant of the Jacobian of the
            transformation.
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, px_1: np.ndarray, px_2: np.ndarray, cond: np.ndarray) -> (np.ndarray, float):
        """Transform (normalize) samples from the target distribution back to
         the prior distribution (usually a multivariate Gaussian).

        Parameters
        ----------
        px_1: np.ndarray (N, D)
            The probabilities of the first part of the N samples in D dimensions.
        px_2: np.ndarray (M, D)
            The probabilities of the first part of the N samples in D dimensions.

        cond: np.ndarray (N, k)
            The k conditioner values for the N samples.

        Returns
        -------
        pz: np.ndarray(N+M, D)
            the transformed samples to the prior distribution
        Log of the Jacobian determinant : float
            The natural logarithm of the determinant of the Jacobian of the
            transformation.
        """
        pass

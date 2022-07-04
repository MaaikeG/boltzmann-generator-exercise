from .potential_base import Potential


class HarmonicWell(Potential):
    def __call__(self, coords, **kwargs):
        x = coords.T[0]
        x2 = 0.5 * x**2
        pot = x - 6*x2 + x2**2 + 0.5 * coords.T[1]**2
        return self._cutoff(pot)

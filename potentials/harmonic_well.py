from .potential_base import Potential


class HarmonicWell(Potential):
    def potential(self, coords, **kwargs):
        x = coords[:, 0]
        x2 = 0.5 * x**2
        return x - 6*x2 + x2**2 + 0.5 * coords[:, 1]**2

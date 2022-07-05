from .distribution_base import Distribution


class HarmonicWell(Distribution):
    def __init__(self, beta=1):
        super().__init__(beta)

    def potential(self, r):
        x = r.T[0]
        x2 = 0.5 * x**2
        pot = self._beta * (x - 6*x2 + x2**2 + 0.5 * r.T[1]**2)
        return self._cutoff(pot)

    def log_prob(self, r):
        return super().log_prob(r)

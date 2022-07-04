import torch


class BoltzmannGenerator:

    def __init__(self, prior, target, flow):
        self.prior = prior
        self.target = target
        self.flow = flow


    def transform(self, samples, is_inverse=False):
        if is_inverse:
            return self.flow.inverse(samples)
        else:
            return self.flow.forward(samples)


    def evaluate_potentials(self, r, is_inverse=False):
        if is_inverse:
            return self.prior.potential(r)
        else:
            return self.target.potential(r)


    def sample(self, n, compute_weights=False):
        samples = self.prior.sample(n)
        transformed_samples, jac_det_log = self.transform(samples)

        if compute_weights:
            weights = self.compute_sample_weights_log(samples, transformed_samples, jac_det_log)

        return transformed_samples, weights


    def compute_sample_weights_log(self, samples, transformed_samples, jac_det_log):
        log_prob = self.prior.log_prob(samples) - jac_det_log
        log_weights = -self.target.potential(transformed_samples) - log_prob
        return log_weights

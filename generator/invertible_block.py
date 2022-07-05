import torch.nn


class InvertibleBlock(torch.nn.Module):
    """ RealNVP block which transforms one half of the input dimensions,
    conditioned on the other half.

    Parameters
    ----------
    transformer : transformers.Transformer
        the transformer
    which : torch.IntTensor
        which dimensions are to be transformed
    on : torch.IntTensor
        the dimensions on which the transformed dimensions are to be conditioned.
    """
    def __init__(self, transformer, which, on):
        super().__init__()

        self.transformer = transformer

        if len(which) != len(on):
            raise ValueError

        self.which = which
        self.on = on


    def _split(self, samples):
        # split samples into z1 and z2
        set1 = torch.index_select(samples, dim=-1, index=self.which)
        set2 = torch.index_select(samples, dim=-1, index=self.on)
        return set1, set2

    def _recombine(self, set1, set2):
        res = torch.zeros([set1.shape[0], set1.shape[-1] + set2.shape[-1]])
        res[:, self.which] = set1
        res[:, self.on] = set2
        return res

    def forward(self, samples):
        z1, z2 = self._split(samples)
        # pass through transformer with args z1, z2, cond(y)
        x1, jac_det = self.transformer.forward(z1, z2)

        return self._recombine(x1, z2), jac_det


    def inverse(self, samples):
        x1, x2 = self._split(samples)

        # pass through transformer_inverse with args x1, x2 and cond(x2)
        z1, jac_det = self.transformer.inverse(x1, x2)

        return self._recombine(z1, x2), jac_det

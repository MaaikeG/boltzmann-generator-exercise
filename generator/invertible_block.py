import torch.nn


class InvertibleBlock(torch.nn.Module):

    def __init__(self, transformer):
        super(InvertibleBlock, self).__init__()
        self.transformer = transformer

    def forward(self, samples):
        # split samples into z1 and z2
        n = shape(samples)[0]/2
        z1, z2 = torch.split(samples, n)

        # pass through transformer with args z1, z2, cond(y)
        transformed_samples, jac_det = self.transformer.forward(z1, cond)

        return transformed_samples, jac_det


    def inverse(self, samples):
        n = shape(samples)[0] / 2
        x1, x2 = torch.split(samples, n)

        # pass through transformer_inverse with args x1, x2 and cond(x2)
        transformed_samples, jac_det = self.transformer.inverse(x2, cond)

        return transformed_samples, jac_det
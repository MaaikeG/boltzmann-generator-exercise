import torch.nn


class InvertibleBlock(torch.nn.Module):
    def __init__(self, transformer, conditioner):
        self.transformer = transformer
        self.conditioner = conditioner


    def forward(self, samples):
        # split samples into z1 and z2

        # apply conditioner to z2

        # pass through transformer with args z1, z2, cond(y)
        pass

    def inverse(self, samples):
        # split samples into x1 and x2 (same split!)

        # apply conditioner to x2

        # pass through tranformer_inverse with args x1, x2 and cond(x2)
        pass

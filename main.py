import torch.nn.functional

import potentials.gaussian_well
import train
from generator.invertible_block import InvertibleBlock
from generator.transformers.affine_transformer import AffineTransformer
from generator.conditioners.conditioner import Conditioner
from generator.flow import Flow


target_distribution = potentials.gaussian_well.TwoDimensionalDoubleWell()

blocks = [InvertibleBlock(
    transformer=AffineTransformer(
        conditioner=Conditioner(dim_in=2,
                                dims_out=[4, 8, 4, 2],
                                activation=torch.nn.Tanh)
    ))]

flow = Flow(blocks=blocks)

train.train(flow, target_distribution, dim=2)

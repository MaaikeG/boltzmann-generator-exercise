import torch
import torch.nn.functional
import matplotlib.pyplot as plt

import train
import potentials.gaussian_well
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

train.train(flow, target_distribution, dim=2, epochs=1_000)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# generate samples
samples = torch.normal(mean=0, std=1, size=[1000, 2])

with torch.no_grad():
    # transform them to our target distribution
    transformed_samples, jac_log = flow(samples)

ax1.scatter(*samples.T)
ax2.scatter(*transformed_samples.T)
plt.show()

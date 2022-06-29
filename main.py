import torch
import torch.nn.functional
import matplotlib.pyplot as plt

import train
import potentials.harmonic_well
from generator.invertible_block import InvertibleBlock
from generator.transformers.affine_transformer import AffineTransformer
from generator.conditioners.conditioner import Conditioner
from generator.flow import Flow


target_distribution = potentials.harmonic_well.HarmonicWell()

blocks = [InvertibleBlock(
    transformer=AffineTransformer(
        scale_conditioner=Conditioner(dim_in=2,
                                dims_out=[4, 8, 4, 2],
                                activation=torch.nn.Tanh),
        shift_conditioner=Conditioner(dim_in=2,
                                      dims_out=[4, 8, 4, 2],
                                      activation=torch.nn.ReLU)
    ))]

flow = Flow(blocks=blocks)

train.train(flow, target_distribution, dim=2, epochs=100)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))


xs = torch.linspace(-5, 5, 50)
ys = torch.linspace(-8, 8, 50)
X, Y = torch.meshgrid(xs, ys)

# Pack X and Y into a single 3-dimensional array
pos = torch.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
pos = target_distribution.potential(pos.flatten(start_dim=0, end_dim=-2))
ax0.contourf(X, Y, pos.reshape(X.shape), cmap='jet', levels=50)

# generate samples
samples = torch.normal(mean=0, std=1, size=[1000, 2])

with torch.no_grad():
    # transform them to our target distribution
    transformed_samples, jac_log = flow(samples)

ax1.scatter(*samples.T)
ax2.scatter(*transformed_samples.T)

plt.show()

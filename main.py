import torch
import torch.nn.functional
import matplotlib.pyplot as plt

from priors.gaussian_prior import GaussianPrior
import train
import potentials.harmonic_well
from generator.invertible_block import InvertibleBlock
from generator.transformers.affine_transformer import AffineTransformer
from generator.conditioners.conditioner import Conditioner
from generator.flow import Flow
from sampling.mcmc import MCMC

target_distribution = potentials.harmonic_well.HarmonicWell()
torch.random.manual_seed(1)

dims = 2

blocks = []
for i in range(2):
    blocks.extend([InvertibleBlock(which=torch.tensor([0], dtype=torch.long),
                                   on=torch.tensor([1], dtype=torch.long),
                                   transformer=AffineTransformer(
                                       scale_conditioner=Conditioner(dims=[dims // 2, 32, dims // 2],
                                                                     activation=torch.nn.Tanh),
                                       shift_conditioner=Conditioner(dims=[dims // 2, 32, dims // 2],
                                                                     activation=torch.nn.ReLU)
                                   )),
                   InvertibleBlock(which=torch.tensor([1], dtype=torch.long),
                                   on=torch.tensor([0], dtype=torch.long),
                                   transformer=AffineTransformer(
                                       scale_conditioner=Conditioner(dims=[dims // 2, 32, dims // 2],
                                                                     activation=torch.nn.Tanh),
                                       shift_conditioner=Conditioner(dims=[dims // 2, 32, dims // 2],
                                                                     activation=torch.nn.ReLU)
                                   ))
                   ])

flow = Flow(blocks=blocks)

sampler = MCMC(sampling_range=torch.tensor([[-5, 5], [-8, 8]]), n_samples=2000, n_dimensions=2, max_step=1)
samples = sampler.get_trajectory(potential=target_distribution, r_initial=torch.tensor([-3, 0]))
samples = torch.cat([samples, (sampler.get_trajectory(potential=target_distribution, r_initial=torch.tensor([3, 0])))])

prior = GaussianPrior(dim=2)

train.train_by_example(flow, samples, prior, epochs=100, batch_size=1024)
train.train_by_energy(flow, target_distribution=target_distribution, prior=prior, epochs=1000)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

xs = torch.linspace(-5, 5, 50)
ys = torch.linspace(-8, 8, 50)
X, Y = torch.meshgrid(xs, ys)

# -----------------------------
# PLOT THE TARGET DISTRIBUTION
# -----------------------------
# Pack X and Y into a single 3-dimensional array
pos = torch.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
pos = target_distribution(pos.flatten(start_dim=0, end_dim=-2))
ax0.contourf(X, Y, pos.reshape(X.shape), cmap='jet', levels=50)
ax0.set_title('potential')

# generate samples
samples = torch.normal(mean=0, std=1, size=[10_000, 2])

with torch.no_grad():
    # transform them to our target distribution
    transformed_samples, jac_log = flow(samples)
ax1.hist2d(*samples.T.numpy(), bins=50, range=[[-4, 4], [-8, 8]])
ax2.hist2d(*transformed_samples.numpy().T, bins=50, range=[[-4, 4], [-8, 8]])

ax1.set_title('prior (sampled)')
ax2.set_title('transformed')

plt.show()

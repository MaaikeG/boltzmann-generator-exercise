import torch.nn.functional
import matplotlib.pyplot as plt

from generator.boltzmann_generator import BoltzmannGenerator
from potentials.gaussian_prior import GaussianPrior
import train
import potentials.harmonic_well
from generator.invertible_block import InvertibleBlock
from generator.transformers.affine_transformer import AffineTransformer
from generator.conditioners.conditioner import Conditioner
from generator.flow import Flow
from sampling.mcmc import MCMC

target_distribution = potentials.harmonic_well.HarmonicWell(beta=0.5)
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

sampler = MCMC(sampling_range=torch.tensor([[-5, 5], [-8, 8]]), n_samples=1024, n_dimensions=2, max_step=1)
samples = sampler.get_trajectory(potential=target_distribution.potential, r_initial=torch.tensor([-3, 0]))
samples = torch.cat(
    [samples, (sampler.get_trajectory(potential=target_distribution.potential, r_initial=torch.tensor([3, 0])))])

# plt.hist2d(*samples.T.numpy(), bins=50, range=[[-4, 4], [-8, 8]])
# plt.show()

bg = BoltzmannGenerator(flow=flow, prior=GaussianPrior(dim=2), target=target_distribution)

train.train_by_example(bg, samples, epochs=200, batch_size=1024)
train.train_by_energy(bg, epochs=1000)


# generate samples
samples = torch.normal(mean=0, std=1, size=[10_000, 2])

with torch.no_grad():
    samples, weights = bg.sample(10000, True)
n_bins = 20
buckets = torch.linspace(-8, 8, n_bins)
bucket_centers = buckets - (buckets[1] - buckets[0])
binned_samples = torch.bucketize(samples[:, 0].detach(), buckets)

pmf = torch.zeros(n_bins)

for i in range(len(pmf)):
    indices = torch.where(binned_samples == i)
    if len(indices[0]) > 0:
        pmf[i] = -torch.logsumexp(weights[indices].detach(), 0)
    else:
        pmf[i] = 1e10

fig, (ax0, ax1) = plt.subplots(1, 2)
# plot the real potential over x...
xs = torch.linspace(-8, 8, 50)
ax0.plot(xs, torch.exp(bg.target.log_prob(torch.vstack([xs, torch.zeros([50])]).T)), label="Target potential")
ax1.plot(xs, bg.target.potential(torch.vstack([xs, torch.zeros([50])]).T), label="Target potential")

# ... and the computed pmf
ax0.plot(bucket_centers, torch.exp(-pmf), label="BG estimate")
ax1.plot(bucket_centers, pmf, label="BG estimate")
ax0.set_ylim(-30, 30)
ax1.set_ylim(-30, 30)
plt.legend()
plt.show()


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
pos = target_distribution.potential(pos.flatten(start_dim=0, end_dim=-2))
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

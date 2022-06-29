import torch
from torch.optim import SGD


def train(flow, target_distribution, dim, epochs=100, batch_size=256):

    optimizer = SGD(flow.parameters(), lr=0.01)

    for i in range(epochs):
        optimizer.zero_grad()

        # generate samples
        samples = torch.normal([dim, batch_size])

        # transform them to our target distribution
        transformed_samples, jac_log = flow(samples)
        transformed_energies = target_distribution.potential(transformed_samples)

        # find KL divergence between the
        loss = (transformed_energies - jac_log).mean()

        loss.backward()
        optimizer.step()

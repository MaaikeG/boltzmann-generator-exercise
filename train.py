import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import potentials.gaussian_well


def _train(flow, transform_fn, dataloader, target, epochs):
    """Trains a flow so that the transformed distribution matches the target
    distribution"""
    optimizer = Adam(flow.parameters(), lr=0.01)

    for i in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # transform them to our target distribution
            transformed_samples, jac_log = transform_fn(batch)

            transformed_energies = target(transformed_samples)

            # find KL divergence between the
            loss = (transformed_energies - jac_log).mean()

            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"epoch: {i} - loss: {loss.item()}")


def train_by_energy(flow, target_distribution, prior, epochs=100, batch_size=1024):
    # generate samples
    samples = prior.sample(batch_size)
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True, drop_last=True)
    return _train(flow, flow.forward, dataloader, target_distribution, epochs)


def train_by_example(flow, samples, prior, epochs=100, batch_size=128):
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True, drop_last=True)
    return _train(flow, flow.inverse, dataloader, prior, epochs)

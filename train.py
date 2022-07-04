from torch.optim import Adam
from torch.utils.data import DataLoader


def _train(bg, dataloader, epochs, inverse=False):
    """Trains a flow so that the transformed distribution matches the target
    distribution"""
    optimizer = Adam(bg.flow.parameters(), lr=0.01)

    for i in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # transform them to our target distribution
            transformed_samples, jac_log = bg.transform(batch, is_inverse=inverse)
            transformed_energies = bg.evaluate_potentials(transformed_samples, is_inverse=inverse)

            # find KL divergence between the
            loss = (transformed_energies - jac_log).mean()

            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"epoch: {i} - loss: {loss.item()}")


def train_by_energy(bg, epochs=100, batch_size=1024):
    samples = bg.prior.sample(batch_size)
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True, drop_last=True)
    return _train(bg, dataloader, epochs, inverse=False)


def train_by_example(bg, samples, epochs=100, batch_size=128):
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True, drop_last=True)
    return _train(bg, dataloader, epochs, inverse=True)

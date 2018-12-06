import zounds
from zounds.learn import GanExperiment
from torch import nn
from torch.nn import functional as F
import argparse

latent_dim = 100
sample_size = 8192


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(latent_dim, 256)
        n = 256
        self.main = nn.Sequential(
            nn.Conv1d(32, n, 25, 1, padding=12),
            nn.Conv1d(n, n, 25, 1, padding=12),
            nn.Conv1d(n, n, 25, 1, padding=12),
            nn.Conv1d(n, n, 25, 1, padding=12)
        )

        self.final = nn.Conv1d(n, 1, 25, 1, padding=12)

    def forward(self, x):
        x = x.view(-1, latent_dim)
        x = self.linear(x)
        x = x.view(-1, 32, 8)
        x = F.relu(x)

        for m in self.main:
            x = F.upsample(x, scale_factor=4)
            x = m(x)
            x = F.relu(x)

        x = F.upsample(x, scale_factor=4)
        x = self.final(x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        n = 256
        self.n = n
        self.main = nn.Sequential(
            nn.Conv1d(1, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, 32, 25, 4, padding=12),
        )

        self.linear = nn.Linear(n, 1)

    def forward(self, x):
        x = x.view(-1, 1, sample_size)
        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)
        x = x.view(-1, 32 * 8)
        x = self.linear(x)
        return x


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = Generator()
        self.discriminator = Critic()

    def forward(self, x):
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.ObjectStorageSettings(),
        zounds.AppSettings()
    ])
    args = parser.parse_args()

    experiment = GanExperiment(
        'wavegan',
        zounds.InternetArchive('AOC11B'),
        GanPair(),
        latent_dim=latent_dim,
        sample_size=sample_size,
        sample_hop=256,
        n_critic_iterations=5,
        n_samples=int(1e5),
        app_port=8888,
        app_secret=args.app_secret,
        object_storage_username=args.object_storage_username,
        object_storage_api_key=args.object_storage_api_key,
        object_storage_region=args.object_storage_region)
    experiment.run()

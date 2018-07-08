from __future__ import print_function
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import calculate_gain, xavier_normal_


class UpsamplingGenerator(nn.Module):
    def __init__(self, latent_dim, n_channels):
        super(UpsamplingGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.Conv1d(latent_dim, n_channels, 3, 1, padding=1, bias=False),
            nn.Conv1d(n_channels, n_channels, 15, 1, padding=7, bias=False),
            nn.Conv1d(n_channels, n_channels, 25, 1, padding=12, bias=False),
            nn.Conv1d(n_channels, n_channels, 25, 1, padding=12, bias=False),
            nn.Conv1d(n_channels, n_channels, 25, 1, padding=12, bias=False),
            nn.Conv1d(n_channels, n_channels, 25, 1, padding=12, bias=False),
            nn.Conv1d(n_channels, 1, 25, 1, padding=12, bias=False),
        )
        for i, p in enumerate(self.main.parameters()):
            if i < len(self.main) - 1:
                xavier_normal_(
                    p.data, calculate_gain('leaky_relu', 0.2))
            else:
                xavier_normal_(p.data, 1)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        for i, m in enumerate(self.main):
            scale_factor = 4 if i < len(self.main) - 1 else 2
            x = F.upsample(x, scale_factor=scale_factor, mode='nearest')
            x = m(x)
            if i < len(self.main) - 1:
                x = F.leaky_relu(x, 0.2)
        return torch.sum(x, dim=1, keepdim=True)


def check_output_size():
    latent_dim = 128
    network = UpsamplingGenerator(latent_dim=latent_dim, n_channels=32)
    noise = torch.FloatTensor(3, latent_dim, 1).normal_(0, 1)
    output = network(noise)
    print('Output shape:', output.shape)
    assert (3, 1, 8192) == output.shape


if __name__ == '__main__':
    check_output_size()

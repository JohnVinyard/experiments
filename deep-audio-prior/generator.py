from torch import nn
from zounds.learn import GatedConvLayer
from torch.nn import functional as F
import torch


class MultiResolutionBlock(nn.Module):
    """
    A layer that convolves several different filter/kernel sizes with the same
    input features
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super(MultiResolutionBlock, self).__init__()
        layers = [
            GatedConvLayer(in_channels, out_channels, k, stride, padding=k // 2)
            for k in kernel_sizes]
        self.main = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return [m(x) for m in self.main]


def sentinel_iter(x):
    """
    Return a tuple of (is_last_value, item) for each item in x
    """
    return ((i == (len(x) - 1), y) for i, y in enumerate(x))


class Generator(nn.Module):
    def __init__(self, latent_dim, upsample_cls, kernel_sizes):
        super(Generator, self).__init__()

        n_filters = 128
        kernel_sizes = kernel_sizes
        nk = len(kernel_sizes)
        self.upsampling_factor = 4
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 256)
        self.gate = nn.Linear(latent_dim, 256)

        self.main = nn.Sequential(
            MultiResolutionBlock(32, n_filters, kernel_sizes),
            MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
            MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
            MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
            MultiResolutionBlock(n_filters * nk, 1, kernel_sizes)
        )

        self.upsamplers = nn.Sequential(*[
            upsample_cls(l.in_channels, self.upsampling_factor)
            for l in self.main])

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        g = self.gate(x)
        a = self.linear(x)
        a = a.view(-1, 32, 8)
        g = g.view(-1, 32, 8)
        x = F.tanh(a) * F.sigmoid(g)

        layer_pairs = zip(self.upsamplers, self.main)
        for is_last_layer, layer_pair in sentinel_iter(layer_pairs):
            upsample, layer = layer_pair
            upsampled = upsample(x)
            bands = layer(upsampled)
            if is_last_layer:
                x = sum(bands)
            else:
                x = torch.cat(bands, dim=1)

        return x

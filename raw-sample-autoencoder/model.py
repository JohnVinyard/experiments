import zounds
from torch import nn
from torch.nn import functional as F
import torch
from zounds.spectral import fir_filter_bank
from torch.autograd import Variable
from scipy.signal import gaussian


# TODO: Factor this into zounds
class MultiResolutionBlock(nn.Module):
    """
    A layer that convolves several different filter/kernel sizes with the same
    input features
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            stride=1,
            activation=F.elu):
        super(MultiResolutionBlock, self).__init__()
        self.activation = activation
        layers = [
            self._make_layer(in_channels, out_channels, k, stride)
            for k in kernel_sizes]
        self.main = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _make_layer(
            self,
            in_channels,
            out_channels,
            kernel,
            stride):
        return nn.Conv1d(
            in_channels, out_channels, kernel, stride, padding=kernel // 2)

    def forward(self, x):
        x = torch.cat([m(x) for m in self.main], dim=1)
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, window_size, kernel_sizes, n_filters):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        ks = kernel_sizes
        nk = len(ks)
        self.nk = nk
        activation = lambda x: F.leaky_relu(x, 0.2)

        self.main = nn.Sequential(
            MultiResolutionBlock(1, n_filters, ks, stride=4, activation=activation),
            MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4, activation=activation),
            MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4, activation=activation),
            MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4, activation=activation),
            MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4, activation=activation),
            MultiResolutionBlock(n_filters * nk, n_filters // nk, ks, stride=2, activation=activation),
        )
        self.final = nn.Linear(n_filters, self.latent_dim)


    def forward(self, x, return_features=False):
        x = x.view(-1, 1, self.window_size)
        features = []

        for m in self.main:
            x = m(x)
            if x.shape[1] == (self.n_filters * self.nk):
                features.append(x)

        x = x.view(-1, self.n_filters)
        x = self.final(x)

        return torch.cat(features, dim=-1) if return_features else x


# class Analyzer(nn.Module):
#     def __init__(self, latent_dim, window_size, kernel_sizes, n_filters):
#         super(Analyzer, self).__init__()
#         self.window_size = window_size
#         self.latent_dim = latent_dim
#         ks = kernel_sizes
#         nk = len(ks)
#
#         activation = F.elu
#
#         self.main = nn.Sequential(
#             MultiResolutionBlock(1, n_filters, ks, stride=4,
#                                  activation=activation),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4,
#                                  activation=activation),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4,
#                                  activation=activation),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=4,
#                                  activation=activation),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks, stride=2,
#                                  activation=activation),
#         )
#
#     def forward(self, x):
#         x = x.view(-1, 1, self.window_size)
#         features = []
#         for m in self.main:
#             x = m(x)
#             norms = torch.norm(x, dim=1, keepdim=True)
#             x = x / (norms + 1e-8)
#             # normed = x / (norms + 1e-8)
#             # normed = normed.transpose(1, 2).contiguous()
#             # normed = normed.view(-1, normed.shape[-1])
#             features.append(x)
#         return torch.cat(features, dim=-1)


class Decoder(nn.Module):
    def __init__(self, latent_dim, kernel_sizes, n_filters):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        ks = kernel_sizes
        nk = len(ks)

        self.main = nn.Sequential(
            MultiResolutionBlock(latent_dim, n_filters, ks),
            MultiResolutionBlock(n_filters * nk, n_filters, ks),
            MultiResolutionBlock(n_filters * nk, n_filters, ks),
            MultiResolutionBlock(n_filters * nk, n_filters, ks),
            MultiResolutionBlock(n_filters * nk, n_filters, ks),
            MultiResolutionBlock(n_filters * nk, 1, ks, activation=lambda x: x),
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        for i, m in enumerate(self.main):
            scale_factor = 2 if i == 0 else 4
            x = F.upsample(x, scale_factor=scale_factor, mode='nearest')
            x = m(x)

        x = torch.sum(x, dim=1, keepdim=True)
        return x


# class DecoderWithFrozenFinalLayer(nn.Module):
#     def __init__(self, latent_dim, kernel_sizes, n_filters):
#         super(DecoderWithFrozenFinalLayer, self).__init__()
#         self.latent_dim = latent_dim
#         ks = kernel_sizes
#         nk = len(ks)
#
#         samplerate = zounds.SR11025()
#
#         scale = zounds.BarkScale(
#             zounds.FrequencyBand(20, samplerate.nyquist - 300), 500)
#         basis = fir_filter_bank(
#             scale, 512, samplerate, gaussian(100, 3))
#
#         weights = Variable(torch.from_numpy(basis).float())
#         # out channels x in channels x kernel width
#         self.weights = weights.view(len(scale), 1, 512).contiguous().cuda()
#
#         self.main = nn.Sequential(
#             MultiResolutionBlock(latent_dim, n_filters, ks),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks),
#             MultiResolutionBlock(n_filters * nk, n_filters, ks),
#             MultiResolutionBlock(n_filters * nk, weights.shape[0] // nk, ks),
#         )
#
#     def forward(self, x):
#         x = x.view(-1, self.latent_dim, 1)
#         for i, m in enumerate(self.main):
#             scale_factor = 2 if i == 0 else 4
#             x = F.upsample(x, scale_factor=scale_factor, mode='nearest')
#             x = m(x)
#
#         # x = torch.sum(x, dim=1, keepdim=True)
#         x = F.conv_transpose1d(x, self.weights)
#         return x[..., :2048]


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, window_size, kernel_sizes, n_filters):
        super(AutoEncoder, self).__init__()
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, window_size, kernel_sizes, n_filters)
        self.decoder = Decoder(latent_dim, kernel_sizes, n_filters)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

"""
Conv Blocks
===============
wavenet
conv
multi-res
dilated stack
"""


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


class MultiDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_sizes, activation=F.elu):
        super(MultiDilationBlock, self).__init__()
        self.activation = activation
        kernel_size = 8

        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=d,
                padding=d // 2 + 1)
            for d in dilation_sizes
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []

        for m in self.main:
            z = m(x)
            features.append(z)
            print z.shape

        x = torch.cat(features, dim=1)
        return self.activation(x)


class WaveNetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            filter_activation=F.tanh,
            gate_activation=F.sigmoid):
        super(WaveNetBlock, self).__init__()
        self.gate_activation = gate_activation
        self.filter_activation = filter_activation
        self.filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        f = self.filter(x)
        g = self.filter(x)
        return self.filter_activation(f) * self.gate_activation(g)


class MultiResolutionWaveNetBlock(MultiResolutionBlock):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1,
                 activation=F.elu):
        super(MultiResolutionWaveNetBlock, self).__init__(
            in_channels, out_channels, kernel_sizes, stride, activation)

    def _make_layer(
            self,
            in_channels,
            out_channels,
            kernel,
            stride):
        return WaveNetBlock(
            in_channels, out_channels, kernel, stride, padding=kernel // 2)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=F.elu):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.l = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.l(x)
        return self.activation(x)


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, activation=F.elu):
        super(DilatedBlock, self).__init__()
        self.activation = activation
        dilations = 2 ** np.arange(0, n_layers)
        self.main = nn.Sequential(*(
            [nn.Conv1d(in_channels, out_channels, 2)] +
            [nn.Conv1d(out_channels, out_channels, 2, padding=d // 2 + 1,
                       dilation=d)
             for d in dilations[1:]]
        ))

    def forward(self, x):
        for m in self.main:
            x = m(x)
            x = self.activation(x)
        return x

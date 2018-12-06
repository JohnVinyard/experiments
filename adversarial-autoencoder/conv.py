from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=F.elu,
            batch_norm=False):

        super(ConvBlock, self).__init__()
        self.activation = activation
        self.l = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    @property
    def in_channels(self):
        return self.l.in_channels

    @property
    def out_channels(self):
        return self.l.out_channels

    def forward(self, x):
        x = self.l(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=F.elu,
            batch_norm=False):

        super(ConvTransposeBlock, self).__init__()
        self.activation = activation
        self.l = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    @property
    def in_channels(self):
        return self.l.in_channels

    @property
    def out_channels(self):
        return self.l.out_channels

    def forward(self, x):
        x = self.l(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.bn(x)
        return x

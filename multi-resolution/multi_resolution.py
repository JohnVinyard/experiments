import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class MultiResolutionLayer(nn.Module):
    def __init__(self, op, in_channels, resolutions, filters, stride=4):
        super(MultiResolutionLayer, self).__init__()
        self.in_channels = in_channels
        layers = [
            op(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=res,
                stride=stride)
            for res in resolutions]
        self.main = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = len(resolutions) * filters

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, -1)
        features = [layer(x) for layer in self.main]
        print [f.shape for f in features]
        size = min(f.shape[-1] for f in features)
        features = torch.cat([f[..., :size] for f in features], dim=1)
        return F.elu(features)


class ConvMultiResolutionLayer(MultiResolutionLayer):
    def __init__(self, in_channels, resolutions, filters, stride=4):
        super(ConvMultiResolutionLayer, self).__init__(
            nn.Conv1d,
            in_channels,
            resolutions,
            filters,
            stride)


class ConvTransposeMultiResolutionLayer(MultiResolutionLayer):
    def __init__(self, in_channels, resolutions, filters, stride=4):
        super(ConvTransposeMultiResolutionLayer, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            resolutions,
            filters,
            stride)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        filters = 32
        self.resolutions = [128, 64, 32, 16]
        self.main = nn.Sequential(
            ConvMultiResolutionLayer(1, self.resolutions, filters, stride=4),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8),
            ConvMultiResolutionLayer(
                filters, self.resolutions, filters, stride=8))

        self.linear_input = len(self.resolutions) * filters
        self.linear = nn.Linear(self.linear_input, 1)

    def forward(self, x):
        x = x.view(-1, 1, 8192)
        for m in self.main:
            x = m(x)
            print x.shape
        x = x.view(-1, self.linear_input)
        x = self.linear(x)
        return x


latent_dim = 128


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        filters = 32
        self.resolutions = [128, 64, 32, 16]
        self.main = nn.Sequential(
            ConvTransposeMultiResolutionLayer(
                latent_dim, self.resolutions, filters, stride=1),
            ConvTransposeMultiResolutionLayer(
                filters, self.resolutions, filters, stride=1),
            ConvTransposeMultiResolutionLayer(
                filters, self.resolutions, filters, stride=1),
            ConvTransposeMultiResolutionLayer(
                filters, self.resolutions, filters, stride=1),
            ConvTransposeMultiResolutionLayer(
                filters, self.resolutions, filters, stride=1),
            ConvTransposeMultiResolutionLayer(
                filters, self.resolutions, filters, stride=1))

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        for m in self.main:
            x = m(x)
            print x.shape
        return x


if __name__ == '__main__':
    c = Critic()
    t = torch.FloatTensor(12, 1, 8192)
    v = Variable(t)
    output = c(v)
    print output.shape

    latent = torch.FloatTensor(12, latent_dim, 1)
    g = Generator()
    v = Variable(latent)
    generated = g(v)
    print generated.shape


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import calculate_gain, xavier_normal
from zounds.learn import DctTransform

from conv import ConvBlock, ConvTransposeBlock
import math

def initialize(module, activation='leaky_relu', param=0.2, final_linear=True):
    for i, p in enumerate(module.parameters()):
        if p.data.dim() < 2:
            continue
        if i == len(module) - 1 and final_linear:
            gain = 1
        else:
            gain = calculate_gain(activation, param)
        p.data = xavier_normal(p.data, gain)


class UpSamplingStack(nn.Module):
    def __init__(
            self,
            full_size,
            factor,
            latent_dim,
            n_filters,
            dct_resample,
            activation=lambda x: F.leaky_relu(x, 0.2),
            kernel_size=25,
            padding=12,
            out_channels=1,
            final_activation=lambda x: x,
            input_size=1,
            zero_slice=None,
            gain=1):

        super(UpSamplingStack, self).__init__()
        self.gain = gain
        self.zero_slice = zero_slice
        self.input_size = input_size
        self.final_activation = final_activation
        self.out_channels = out_channels
        self.dct_resample = dct_resample
        self.activation = activation
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.factor = factor
        self.full_size = full_size

        size = int(full_size * factor)
        n_layers = int(np.log2(size) - np.log2(input_size))
        layers = []

        for i in xrange(n_layers):
            layer = ConvTransposeBlock(
                in_channels=self.latent_dim if i == 0 else self.n_filters,
                out_channels=self.n_filters,
                kernel_size=kernel_size + 1,
                stride=2,
                padding=padding,
                activation=self.activation
            )
            for p in layer.parameters():
                if p.data.dim() < 2:
                    continue
                p.data = xavier_normal(
                    p.data, calculate_gain('leaky_relu', 0.2))
            layers.append(layer)

        self.main = nn.Sequential(*layers)

        self.final = ConvBlock(
            n_filters,
            # self.out_channels,
            3,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            activation=self.final_activation)
        for p in self.final.parameters():
            if p.data.dim() < 2:
                continue
            p.data = xavier_normal(p.data, 1)

    def forward(self, x, upsample=True):
        # batch x 6 x 16
        x = x.view(x.shape[0], self.latent_dim, -1)

        x = self.main(x)
        x = self.final(x)

        if upsample:
            x = x * self.gain
            x = self.dct_resample.dct_resample(
                x, self.full_size // x.shape[-1], zero_slice=self.zero_slice)
            x = x.view(-1, 1, self.full_size)
        return x


class DownSamplingStack(nn.Module):
    def __init__(
            self,
            full_size,
            factor,
            n_filters,
            activation=lambda x: F.leaky_relu(x, 0.2),
            kernel_size=25,
            padding=12,
            in_channels=1,
            final_size=1,
            out_features=None,
            final_activation=None,
            gain=1):

        super(DownSamplingStack, self).__init__()
        self.gain = gain
        self.final_activation = final_activation or activation
        self.out_features = out_features or n_filters
        self.in_channels = in_channels
        self.activation = activation
        self.n_filters = n_filters
        self.factor = factor
        self.full_size = full_size

        self.size = int(full_size * factor)

        n_layers = int(np.log2(self.size) - np.log2(final_size))
        layers = []
        for i in xrange(n_layers):
            last = i == n_layers - 1
            layer = ConvBlock(
                in_channels=self.in_channels if i == 0 else n_filters,
                out_channels=self.out_channels if last else n_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                activation=self.final_activation if last else activation,
                batch_norm=False)
            layers.append(layer)
            for p in layer.parameters():
                if p.data.dim() < 2:
                    continue
                p.data = xavier_normal(
                    p.data, calculate_gain('leaky_relu', 0.2))

        self.main = nn.Sequential(*layers)

    @property
    def out_channels(self):
        return self.out_features

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.size)
        x = self.gain * x
        x = self.main(x)
        return x


class MultiScaleEncoder2(nn.Module):
    def __init__(self, latent_dim, window_size, n_filters, gains):
        super(MultiScaleEncoder2, self).__init__()
        self.gains = gains
        self.n_filters = n_filters
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.scales = [1, 0.5, 0.25, 0.125, 0.0625][::-1]
        self.dct = DctTransform(use_cuda=True)

        kernel = 25
        padding = 12
        final_size = [16] * 5
        out_features = [6] * 5

        self.main = nn.Sequential(
            DownSamplingStack(
                window_size, 0.0625, n_filters, kernel_size=kernel,
                padding=padding, final_size=final_size[0],
                out_features=out_features[0],
                final_activation=lambda x: x, gain=gains[0]),
            DownSamplingStack(
                window_size, 0.125, n_filters, kernel_size=kernel,
                padding=padding, final_size=final_size[0],
                out_features=out_features[1],
                final_activation=lambda x: x, gain=gains[1]),
            DownSamplingStack(
                window_size, 0.25, n_filters, kernel_size=kernel,
                padding=padding, final_size=final_size[0],
                out_features=out_features[2],
                final_activation=lambda x: x, gain=gains[2]),
            DownSamplingStack(
                window_size, 0.5, n_filters, kernel_size=kernel,
                padding=padding, final_size=final_size[0],
                out_features=out_features[3],
                final_activation=lambda x: x, gain=gains[3]),
            DownSamplingStack(
                window_size, 1.0, n_filters, kernel_size=kernel,
                padding=padding, final_size=final_size[0],
                out_features=out_features[4],
                final_activation=lambda x: x, gain=gains[4]),
        )

    def forward(self, x, return_features=False):
        x = x.view(-1, 1, self.window_size)
        bands = self.dct.frequency_decomposition(x, self.scales)

        latents = []
        for band, m in zip(bands, self.main):
            latent = m(band)
            latents.append(latent)

        # concatenate several batch x 96 x 1 features, which expands to
        # batch x 30 x 16 or batch x 5 x 6 x 16
        x = torch.cat(latents, dim=1)
        return x
        # return x.view(-1, self.latent_dim, 1)


class MultiScaleFlatEncoder(MultiScaleEncoder2):
    def __init__(self, latent_dim, window_size, n_filters, gains):
        super(MultiScaleFlatEncoder, self).__init__(
            latent_dim, window_size, n_filters, gains)
        self.final = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 128),
        )
        initialize(self.final, final_linear=True)

    def forward(self, x, return_features=False):
        x = super(MultiScaleFlatEncoder, self).forward(
            x, return_features=return_features)
        x = x.view(-1, self.latent_dim)
        for i, f in enumerate(self.final):
            x = f(x)
            if i < len(self.final) - 1:
                x = F.leaky_relu(x, 0.2)

        x = F.relu(x)
        summed = torch.sum(x, dim=1, keepdim=True)
        x = x / summed
        return x


class MultiscaleCritic(MultiScaleEncoder2):
    def __init__(self, latent_dim, window_size, n_filters, gains):
        super(MultiscaleCritic, self).__init__(
            latent_dim, window_size, n_filters, gains)

        self.loss = nn.Linear(self.latent_dim, 1, bias=False)
        for p in self.loss.parameters():
            if p.data.dim() < 2:
                continue
            p.data = xavier_normal(p.data, 1)

    def forward(self, x, return_features=False):
        x = super(MultiscaleCritic, self).forward(
            x, return_features=return_features)
        if return_features:
            return x
        else:
            x = x.view(-1, self.loss.in_features)
            return self.loss(x)


class PerceptualCritic(nn.Module):
    def __init__(self, latent_dim, window_size, n_filters):
        super(PerceptualCritic, self).__init__()
        self.n_filters = n_filters
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.perceptual_model = None

        self.main = nn.Sequential(
            nn.Conv1d(256, n_filters, 1, 1, 0, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 5, 1, 0, bias=False),
        )

        for p in self.main.parameters():
            if p.data.dim() < 2:
                continue
            p.data = xavier_normal(p.data, calculate_gain('leaky_relu', 0.2))

        self.final = nn.Linear(n_filters, 1, bias=False)

        for p in self.final.parameters():
            if p.data.dim() < 2:
                continue
            p.data = xavier_normal(p.data, 1)

    def forward(self, x, return_features=False):
        batch_size = x.shape[0]
        x = self.perceptual_model._transform(x)
        x = x.view(batch_size, 256, -1)

        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)

        x = x.view(-1, self.final.in_features)
        if return_features:
            return x

        x = self.final(x)

        return x



class MultiScaleGenerator2(nn.Module):
    def __init__(self, latent_dim, window_size, n_filters, gains):
        super(MultiScaleGenerator2, self).__init__()
        self.gains = gains
        self.n_filters = n_filters
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.scales = [1, 0.5, 0.25, 0.125, 0.0625]
        self.dct = DctTransform(use_cuda=True)

        kernel = 25
        padding = 12
        final_activation = lambda x: x
        out_channels = 1
        input_size = [16] * 5
        input_features = [6] * 5


        self.main = nn.Sequential(
            UpSamplingStack(
                window_size, 0.0625, input_features[0], n_filters, self.dct,
                kernel_size=kernel, padding=padding, out_channels=out_channels,
                final_activation=final_activation, input_size=input_size[0],
                zero_slice=None, gain=gains[0]),
            UpSamplingStack(
                window_size, 0.125, input_features[1], n_filters, self.dct,
                kernel_size=kernel, padding=padding, out_channels=out_channels,
                final_activation=final_activation, input_size=input_size[1],
                zero_slice=slice(0, 32), gain=gains[1]
            ),
            UpSamplingStack(
                window_size, 0.25, input_features[2], n_filters, self.dct,
                kernel_size=kernel, padding=padding, out_channels=out_channels,
                final_activation=final_activation, input_size=input_size[2],
                zero_slice=slice(0, 64), gain=gains[2]
            ),
            UpSamplingStack(
                window_size, 0.5, input_features[3], n_filters, self.dct,
                kernel_size=kernel, padding=padding, out_channels=out_channels,
                final_activation=final_activation, input_size=input_size[3],
                zero_slice=slice(0, 128), gain=gains[3]
            ),
            UpSamplingStack(
                window_size, 1.0, input_features[4], n_filters, self.dct,
                kernel_size=kernel, padding=padding, out_channels=out_channels,
                final_activation=final_activation, input_size=input_size[4],
                zero_slice=slice(0, 256), gain=gains[4]
            )
        )

        self.slices = []
        for f, l in zip(input_features, input_size):
            start = 0 if not len(self.slices) else self.slices[-1].stop
            stop = start + f
            self.slices.append(slice(start, stop))

        # import zounds
        # window = zounds.OggVorbisWindowingFunc() * np.ones(window_size)
        # self.window = torch.from_numpy(window) \
        #     .float().cuda().view(1, 1, window_size)

    def forward(self, x, sum_bands=True):
        batch = x.shape[0]

        x = x.view(batch, -1, 16)

        signals = []

        for m, sl in zip(self.main, self.slices):
            # batch x 96 x 1
            signal = m(x[:, sl, :], upsample=sum_bands)
            signals.append(signal)

        if sum_bands:
            x = torch.cat(signals, dim=1)
            x = torch.sum(x, dim=1, keepdim=True)
            return x #* self.window
        else:
            return signals


class MultiScaleFlatGenerator(MultiScaleGenerator2):
    def __init__(self, latent_dim, window_size, n_filters, gains):
        super(MultiScaleFlatGenerator, self).__init__(
            latent_dim, window_size, n_filters, gains)
        self.initial = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 256),
            nn.Linear(256, self.latent_dim)
        )
        initialize(self.initial, final_linear=False)

    def forward(self, x, sum_bands=True):
        x = x.view(-1, 128)
        for i, f in enumerate(self.initial):
            x = f(x)
            if i < len(self.initial) - 1:
                x = F.leaky_relu(x, 0.2)
        x = super(MultiScaleFlatGenerator, self).forward(x, sum_bands=sum_bands)
        return x


class LatentCritic(nn.Module):
    def __init__(self, latent_dim):
        super(LatentCritic, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
        )
        for i, p in enumerate(self.main.parameters()):
            if p.data.dim() < 2:
                continue
            gain = \
                1 if i == len(self.main) - 1 \
                    else calculate_gain('leaky_relu', 0.2)
            p.data = xavier_normal(p.data, gain)

    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        for i, m in enumerate(self.main):
            x = m(x)
            if i < len(self.main) - 1:
                x = F.leaky_relu(x, 0.2)
        return x


class Network(nn.Module):
    def __init__(
            self,
            latent_dimension,
            n_filters,
            window_size,
            gains):
        super(Network, self).__init__()

        self.gains = gains
        self.window_size = window_size
        self.n_filters = n_filters
        self.latent_dimension = latent_dimension
        self.latent_critic = LatentCritic(latent_dimension)
        self.encoder = MultiScaleFlatEncoder(
            latent_dimension, window_size, n_filters,
            [1. / g for g in self.gains])
        # self.encoder = MultiScaleEncoder2(
        #     latent_dimension, window_size, n_filters,
        #     [1. / g for g in self.gains])
        self.generator = MultiScaleFlatGenerator(
            latent_dimension, window_size, n_filters, gains)
        # self.generator = MultiScaleGenerator2(
        #     latent_dimension, window_size, n_filters, gains)
        # self.data_critic = MultiscaleCritic(
        #     latent_dimension, window_size, n_filters,
        #     [1. / g for g in self.gains])
        self.data_critic = PerceptualCritic(
            latent_dimension, window_size, n_filters)

    @property
    def discriminator(self):
        return self.data_critic


def assert_shape(expected, actual, name):
    assert \
        expected == actual.shape, \
        'for {name}, expected shape {expected}, but was {actual.shape}'.format(
            **locals())
    print '{name} has correct shape {expected}'.format(**locals())


def test_networks():
    import torch
    from torch.autograd import Variable
    import zounds
    from zounds.learn import PerceptualLoss

    samplerate = zounds.SR11025()

    scale = zounds.BarkScale(
        zounds.FrequencyBand(20, samplerate.nyquist - 300), 512)

    loss = PerceptualLoss(
        scale,
        samplerate,
        lap=1,
        log_factor=10,
        basis_size=512,
        frequency_weighting=zounds.AWeighting(),
        cosine_similarity=False).cuda()

    latent_dim = 72
    n_filters = 32
    window_size = 4096
    network = Network(latent_dim, n_filters, window_size,
                      [1, 1, 1, 1, 1]).cuda()
    network.data_critic.perceptual_model = loss
    batch_size = 3

    data = Variable(torch.FloatTensor(*(batch_size, 1, window_size))).cuda()
    assert_shape((batch_size, 1, window_size), data, 'input data')

    latent = network.encoder(data)
    assert_shape((batch_size, latent_dim, 1), latent, 'latent code')

    generated = network.generator(latent)
    assert_shape((batch_size, 1, window_size), generated, 'generated data')

    # latent_w = network.latent_critic(latent)
    # assert_shape((batch_size, 1), latent_w, 'latent_w')

    data_w = network.data_critic(generated)
    assert_shape((batch_size, 1), data_w, 'data_w')


if __name__ == '__main__':
    test_networks()

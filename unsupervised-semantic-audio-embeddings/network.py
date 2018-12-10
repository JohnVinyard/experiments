import numpy as np
from scipy.signal import morlet
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
from zounds.learn.util import batchwise_unit_norm
import zounds


def make_filter_bank(samplerate, basis_size, kernel_size, scale):
    """
    Create a bank of finite impulse response filters, with
    frequencies centered on the sub-bands of scale
    """
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    for i, band in enumerate(scale):
        # cycles determines the tradeoff between time and frequency
        # resolution.  We'd like good frequency resolution for lower
        # frequencies, and good time resolution for higher ones
        cycles = max(64, int(samplerate) / band.center_frequency)
        basis[i] = morlet(
            kernel_size,  # wavelet size
            cycles,  # time-frequency resolution tradeoff
            (band.center_frequency / samplerate.nyquist))  # frequency
    return basis.real


def batchwise_mean_std_normalization(x):
    orig_shape = x.shape
    x = x.view(x.shape[0], -1)
    x = x - x.mean(dim=1, keepdim=True)
    x = x / (x.std(dim=1, keepdim=True) + 1e-8)
    x = x.view(orig_shape)
    return x


class FilterBank(nn.Module):
    def __init__(self, samplerate, channels, kernel_size, scale):
        super(FilterBank, self).__init__()
        filter_bank = make_filter_bank(samplerate, channels, kernel_size, scale)
        self.scale = scale
        self.filter_bank = torch.from_numpy(filter_bank).float() \
            .view(len(scale), 1, kernel_size)
        self.filter_bank.requires_grad = False

    def to(self, *args, **kwargs):
        self.filter_bank = self.filter_bank.to(*args, **kwargs)
        return super(FilterBank, self).to(*args, **kwargs)

    def convolve(self, x):
        x = x.view(-1, 1, x.shape[-1])
        x = F.conv1d(
            x, self.filter_bank, padding=self.filter_bank.shape[-1] // 2)
        return x

    def log_magnitude(self, x):
        x = F.relu(x)
        x = 20 * torch.log10(1 + x)
        return x

    def temporal_pooling(self, x, kernel_size, stride):
        x = F.avg_pool1d(x, kernel_size, stride, padding=kernel_size // 2)
        return x

    def normalize(self, x):
        """
        give each instance zero mean and unit variance
        """
        orig_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        x = x.view(orig_shape)
        return x

    def transform(self, samples, pooling_kernel_size, pooling_stride):
        # convert the raw audio samples to a PyTorch tensor
        tensor_samples = torch.from_numpy(samples).float() \
            .to(self.filter_bank.device)

        # compute the transform
        spectral = self.convolve(tensor_samples)
        log_magnitude = self.log_magnitude(spectral)
        pooled = self.temporal_pooling(
            log_magnitude, pooling_kernel_size, pooling_stride)

        # convert back to an ArrayWithUnits instance
        samplerate = samples.samplerate
        time_frequency = pooled.data.cpu().numpy().squeeze().T
        time_frequency = zounds.ArrayWithUnits(time_frequency, [
            zounds.TimeDimension(
                frequency=samplerate.frequency * pooling_stride,
                duration=samplerate.frequency * pooling_kernel_size),
            zounds.FrequencyDimension(self.scale)
        ])
        return time_frequency

    def forward(self, x, normalize=True):
        nsamples = x.shape[-1]
        x = self.convolve(x)
        x = self.log_magnitude(x)

        if normalize:
            x = self.normalize(x)

        return x[..., :nsamples].contiguous()


class EmbeddingNetwork(nn.Module):
    """
    Compute Log-scaled mel spectrogram, followed by a vanilla 2d convolutional
    network with alternating convolutional and average pooling layers
    """

    def __init__(self):
        super(EmbeddingNetwork, self).__init__()

        frequency_channels = 128
        channels = frequency_channels

        sr = zounds.SR11025()
        band = zounds.FrequencyBand(20, sr.nyquist)
        scale = zounds.MelScale(band, frequency_channels)
        self.bank = FilterBank(sr, frequency_channels, 512, scale)

        self.main = nn.Sequential(
            nn.Conv2d(1, channels, (13, 3), padding=(7, 1), bias=False),
            nn.MaxPool2d((2, 2), (2, 2), padding=(1, 1)),
            nn.Conv2d(channels, channels, (13, 3), padding=(7, 1), bias=False),
            nn.MaxPool2d((2, 2), (2, 2), padding=(1, 1)),
            nn.Conv2d(channels, channels, (13, 3), padding=(7, 1), bias=False),
            nn.MaxPool2d((2, 2), (2, 2), padding=(1, 1)),
            nn.Conv2d(channels, channels, (13, 3), padding=(7, 1), bias=False),
            nn.MaxPool2d((2, 2), (2, 2), padding=(1, 1)),
        )

        self.linear = nn.Linear(128, 128, bias=False)

    def trainable_parameter_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def to(self, *args, **kwargs):
        self.bank = self.bank.to(*args, **kwargs)
        return super(EmbeddingNetwork, self).to(*args, **kwargs)

    def initialize_weights(self):
        for m in self.main.parameters():
            if m.data.dim() > 2:
                xavier_normal_(m.data, calculate_gain('leaky_relu', 0.2))

        for m in self.linear.parameters():
            if m.data.dim() > 2:
                xavier_normal_(m.data, 1)

    def forward(self, x):
        # normalize
        x = x.view(-1, 8192)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        x = x.view(-1, 1, 8192)

        # filter bank
        x = self.bank(x, normalize=False)

        # temporal pooling
        x = F.avg_pool1d(x, 128, 64, padding=64)

        # give zero mean and unit variance
        x = batchwise_mean_std_normalization(x)

        # view as 2d spectrogram "image", so dimension are now
        # (batch, 1, 128, 128)
        x = x[:, None, ...]

        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)

        # global max pooling
        x = F.max_pool2d(x, x.shape[2:])

        # linear transformation
        x = x.view(-1, self.linear.in_features)
        x = self.linear(x)
        x = batchwise_unit_norm(x)
        return x

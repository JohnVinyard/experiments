import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from upsample import \
    LinearUpSamplingBlock, LearnedUpSamplingBlock, DctUpSamplingBlock, \
    NearestNeighborUpsamplingBlock
from conv import \
    ConvBlock, MultiResolutionBlock, WaveNetBlock, DilatedBlock, \
    MultiResolutionWaveNetBlock, MultiDilationBlock
from tosamples import SumToSamples, EnsureSize
from zounds.learn import DctTransform
import numpy as np
import math


class BaseGenerator(nn.Module):
    def __init__(self, latent_dim, window_size, layers):
        super(BaseGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(*layers)
        self.window_size = window_size
        self.parameter_count = sum(
            np.prod(p.data.shape) for p in self.parameters())

    def cuda(self, device=None):
        return super(BaseGenerator, self).cuda(device=device)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    def _extra_display_data(self):
        return dict()

    def __repr__(self):
        d = dict(cls=self.__class__.__name__, params=self.parameter_count)
        d.update(self._extra_display_data())
        return str(d)

    def __str__(self):
        return self.__repr__()


class ConvGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters):
        super(ConvGenerator, self).__init__(latent_dim, window_size, [
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12,
                                   activation=lambda x: x),
            SumToSamples()
        ])
        self.n_filters = n_filters


class HybridConvGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters):
        super(HybridConvGenerator, self).__init__(latent_dim, window_size, [
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12,
                                   activation=lambda x: x),
            SumToSamples()
        ])
        self.n_filters = n_filters

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        for layer in self.layers:
            nx = layer(x)
            if x.shape[1] == nx.shape[1]:
                factor = nx.shape[-1] // x.shape[-1]
                x = nx + F.upsample(x, scale_factor=factor, mode='nearest')
            else:
                x = nx
        return x


class FrozenFinalLayer(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, final_weights):
        super(FrozenFinalLayer, self).__init__(latent_dim, window_size, [
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, final_weights.shape[0], 26, 2, 12,
                                   F.elu),
        ])
        self.final_weights = final_weights
        self.n_filters = n_filters

    def forward(self, x):
        x = super(FrozenFinalLayer, self).forward(x)
        x = F.conv_transpose1d(x, self.final_weights)
        return x[..., :self.window_size].contiguous()


class UpSamplingGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(UpSamplingGenerator, self).__init__(latent_dim, window_size, [
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 3, 1, 1),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 15, 1, 7),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, activation=lambda x: x),
            SumToSamples()
        ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class DeepUpSamplingGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(DeepUpSamplingGenerator, self).__init__(latent_dim, window_size, [
            upsample_cls(2),
            ConvBlock(latent_dim, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 3, 1, 1),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 3, 1, 1, lambda x: x),
            SumToSamples()
        ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class PschyoAcousticUpsamplingGenerator(BaseGenerator):
    """
    Only begin to use unlearned upsampling at the point where we enter the
    range of human hearing
    """

    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(PschyoAcousticUpsamplingGenerator, self).__init__(
            latent_dim, window_size, [
                LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
                upsample_cls(2),
                ConvBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(2),
                ConvBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(2),
                ConvBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(2),
                ConvBlock(
                    n_filters, 1, 25, 1, 12, activation=lambda x: x),
                SumToSamples()
            ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class PsychoAcousticMultiResUpSamplingGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            kernel_sizes):
        ks = kernel_sizes
        nk = len(ks)

        super(PsychoAcousticMultiResUpSamplingGenerator, self).__init__(
            latent_dim, window_size, [
                LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
                upsample_cls(2),
                MultiResolutionBlock(n_filters, n_filters, ks),
                upsample_cls(2),
                MultiResolutionBlock(n_filters * nk, n_filters, ks),
                upsample_cls(2),
                MultiResolutionBlock(n_filters * nk, n_filters, ks),
                upsample_cls(2),
                MultiResolutionBlock(n_filters * nk, 1, ks,
                                     activation=lambda x: x),
                SumToSamples()
            ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class PsychoAcousticMultiDilationUpSamplingGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            dilation_sizes):
        ks = dilation_sizes
        nk = len(ks)

        super(PsychoAcousticMultiDilationUpSamplingGenerator, self).__init__(
            latent_dim, window_size, [
                LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
                LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
                upsample_cls(2),
                MultiDilationBlock(n_filters, n_filters, ks),
                upsample_cls(2),
                MultiDilationBlock(n_filters * nk, n_filters, ks),
                upsample_cls(2),
                MultiDilationBlock(n_filters * nk, n_filters, ks),
                upsample_cls(2),
                MultiDilationBlock(n_filters * nk, 1, ks,
                                   activation=lambda x: x),
                SumToSamples(),
                EnsureSize(window_size)
            ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class MultiScaleGenerator(nn.Module):
    def __init__(self, latent_dim, window_size, n_filters):
        super(MultiScaleGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.window_size = window_size
        self.dct = DctTransform()

        self._512 = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, lambda x: x)
        )

        self._1024 = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, lambda x: x),
        )

        self._2048 = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, lambda x: x),
        )

        self._4096 = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, lambda x: x),
        )

        self._8192 = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 2, 12, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, lambda x: x),
        )

        self.scales = \
            [self._512, self._1024, self._2048, self._4096, self._8192]
        self.parameter_count = sum(
            np.prod(p.data.shape) for p in self.parameters())

    def cuda(self, device=None):
        self.dct = self.dct.cuda()
        return super(MultiScaleGenerator, self).cuda(device=device)

    def __repr__(self):
        d = dict(cls=self.__class__.__name__, params=self.parameter_count)
        return str(d)

    def __str__(self):
        return self.__repr__()

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        scales = []
        for scale in self.scales:
            output = scale(x)
            upsampled = self.dct.dct_resample(
                output, self.window_size // output.shape[-1])
            scales.append(upsampled)
        x = torch.cat(scales, dim=1)
        return torch.sum(x, dim=1, keepdim=True)


class MultiScaleUpSamplingGenerator(nn.Module):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(MultiScaleUpSamplingGenerator, self).__init__()

        self.upsample_cls = upsample_cls
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.dct = DctTransform()

        self._512 = nn.Sequential(
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, lambda x: x)
        )

        self._1024 = nn.Sequential(
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, lambda x: x)
        )

        self._2048 = nn.Sequential(
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, lambda x: x)
        )

        self._4096 = nn.Sequential(
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, lambda x: x)
        )

        self._8192 = nn.Sequential(
            upsample_cls(4),
            ConvBlock(latent_dim, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(4),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, n_filters, 25, 1, 12, F.elu),
            upsample_cls(2),
            ConvBlock(n_filters, 1, 25, 1, 12, lambda x: x)
        )

        self.scales = \
            [self._512, self._1024, self._2048, self._4096, self._8192]
        self.parameter_count = sum(
            np.prod(p.data.shape) for p in self.parameters())

    def cuda(self, device=None):
        self.dct = self.dct.cuda()
        return super(MultiScaleUpSamplingGenerator, self).cuda(device=device)

    def __repr__(self):
        d = dict(
            cls=self.__class__.__name__,
            params=self.parameter_count,
            upsample_cls=self.upsample_cls)
        return str(d)

    def __str__(self):
        return self.__repr__()

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        scales = []
        for scale in self.scales:
            output = scale(x)
            upsampled = self.dct.dct_resample(
                output, self.window_size // output.shape[-1])
            scales.append(upsampled)
        x = torch.cat(scales, dim=1)
        return torch.sum(x, dim=1, keepdim=True)


class UpSamplingWavenetGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(UpSamplingWavenetGenerator, self).__init__(
            latent_dim, window_size, [
                upsample_cls(4),
                WaveNetBlock(latent_dim, n_filters, 3, 1, 1),
                upsample_cls(4),
                WaveNetBlock(n_filters, n_filters, 15, 1, 7),
                upsample_cls(4),
                WaveNetBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(4),
                WaveNetBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(4),
                WaveNetBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(4),
                WaveNetBlock(n_filters, n_filters, 25, 1, 12),
                upsample_cls(2),
                WaveNetBlock(n_filters, 1, 25, 1, 12, lambda x: x),
                SumToSamples()
            ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class UpSamplingMultiResGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            kernel_sizes):
        nk = len(kernel_sizes)

        super(UpSamplingMultiResGenerator, self).__init__(
            latent_dim, window_size, [
                upsample_cls(4),
                ConvBlock(latent_dim, n_filters, 3, 1, 1),
                upsample_cls(4),
                ConvBlock(n_filters, n_filters, 15, 1, 7),
                upsample_cls(4),
                MultiResolutionBlock(n_filters, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(2),
                MultiResolutionBlock(
                    n_filters * nk,
                    1,
                    kernel_sizes,
                    activation=lambda x: x),
                SumToSamples()
            ])
        self.kernel_sizes = kernel_sizes
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class UpSamplingGeneratorFinalMultiRes(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls, ks):
        nk = len(ks)

        super(UpSamplingGeneratorFinalMultiRes, self).__init__(latent_dim,
                                                               window_size, [
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       latent_dim,
                                                                       n_filters,
                                                                       3, 1, 1),
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       15, 1,
                                                                       7),
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       25, 1,
                                                                       12),
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       25, 1,
                                                                       12),
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       25, 1,
                                                                       12),
                                                                   upsample_cls(
                                                                       4),
                                                                   ConvBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       25, 1,
                                                                       12),
                                                                   upsample_cls(
                                                                       2),
                                                                   MultiResolutionBlock(
                                                                       n_filters,
                                                                       n_filters,
                                                                       ks,
                                                                       activation=lambda
                                                                           x: x),
                                                                   # ConvBlock(n_filters, 1, 25, 1, 12, activation=lambda x: x),
                                                                   SumToSamples()
                                                               ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class AllMultiResGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            ks,
            inner_activation=F.elu,
            final_activation=lambda x: x,
            delta=False,
            positional_encoding=False):

        nk = len(ks)

        out_channels = n_filters * nk
        if positional_encoding:
            out_channels += 1

        super(AllMultiResGenerator, self).__init__(latent_dim, window_size, [
            upsample_cls(4),
            MultiResolutionBlock(latent_dim, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(out_channels, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(out_channels, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(out_channels, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(out_channels, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(out_channels, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(2),
            MultiResolutionBlock(out_channels, 1, ks,
                                 activation=final_activation),
            SumToSamples()
        ])

        self.delta = delta
        self.upsample_cls = upsample_cls
        self.inner_activation = inner_activation
        self.final_activation = final_activation
        self.positional_encoding = positional_encoding
        self.ks = ks

    def _extra_display_data(self):
        return dict(
            ks=self.ks,
            delta=self.delta,
            positional_encoding=self.positional_encoding,
            upsample=self.upsample_cls.__name__,
            inner_activation=self.inner_activation.__name__,
            final_activation=self.final_activation.__name__)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if self.positional_encoding \
                    and not isinstance(layer, self.upsample_cls) \
                    and not isinstance(layer, SumToSamples):
                z = torch.linspace(0, 1, x.shape[-1])
                z = z.repeat(x.shape[0], 1, 1)
                z = Variable(z).cuda()
                x = torch.cat([x, z], dim=1)

        if self.delta:
            x = torch.cumprod(x, dim=-1)

        return x


class MultiBranch(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            ks,
            inner_activation=F.elu,
            final_activation=lambda x: x):
        nk = len(ks)

        super(MultiBranch, self).__init__(latent_dim, window_size, [
            upsample_cls(4),
            MultiResolutionBlock(latent_dim, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(n_filters * nk, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(n_filters * nk, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(n_filters * nk, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(n_filters * nk, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(4),
            MultiResolutionBlock(n_filters * nk, n_filters, ks,
                                 activation=inner_activation),
            upsample_cls(2),
            MultiResolutionBlock(n_filters * nk, 1, ks,
                                 activation=final_activation),
            SumToSamples()
        ])

        self.amplitude = nn.Sequential(
            LearnedUpSamplingBlock(latent_dim, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, n_filters, 26, 4, 11, F.elu),
            LearnedUpSamplingBlock(n_filters, 1, 26, 2, 12, F.elu),
        )

        self.upsample_cls = upsample_cls
        self.inner_activation = inner_activation
        self.final_activation = final_activation

    def forward(self, x):
        spectral = super(MultiBranch, self).forward(x)
        amp = self.amplitude(x.view(-1, self.latent_dim, 1))
        return spectral * amp

    def _extra_display_data(self):
        return dict(
            upsample=self.upsample_cls.__name__,
            inner_activation=self.inner_activation.__name__,
            final_activation=self.final_activation.__name__)


class LearnedFirstLayerMultiResGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls, ks):
        nk = len(ks)

        super(LearnedFirstLayerMultiResGenerator, self).__init__(latent_dim,
                                                                 window_size, [
                                                                     LearnedUpSamplingBlock(
                                                                         latent_dim,
                                                                         n_filters,
                                                                         4, 4,
                                                                         0),
                                                                     MultiResolutionBlock(
                                                                         n_filters,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         4),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         4),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         4),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         4),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         4),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         n_filters,
                                                                         ks),
                                                                     upsample_cls(
                                                                         2),
                                                                     MultiResolutionBlock(
                                                                         n_filters * nk,
                                                                         1, ks,
                                                                         activation=lambda
                                                                             x: x),
                                                                     SumToSamples()
                                                                 ])

        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class UpSamplingMultiResGeneratorFinalFrozenLayer(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            kernel_sizes,
            final_weights):
        self.final_weights = final_weights
        nk = len(kernel_sizes)

        super(UpSamplingMultiResGeneratorFinalFrozenLayer, self).__init__(
            latent_dim, window_size, [
                upsample_cls(4),
                ConvBlock(latent_dim, n_filters, 3, 1, 1),
                upsample_cls(4),
                ConvBlock(n_filters, n_filters, 15, 1, 7),
                upsample_cls(4),
                MultiResolutionBlock(n_filters, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionBlock(n_filters * nk, n_filters, kernel_sizes),
                upsample_cls(2),
                MultiResolutionBlock(
                    n_filters * nk,
                    self.final_weights.shape[0] // nk,
                    kernel_sizes,
                    activation=lambda x: x),
            ])
        self.kernel_sizes = kernel_sizes
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)

    def forward(self, x):
        x = super(UpSamplingMultiResGeneratorFinalFrozenLayer, self).forward(x)
        return F.conv_transpose1d(x, self.final_weights)[...,
               :self.window_size].contiguous()


class UpSamplingMultiDilationGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            dilation_sizes):
        nd = len(dilation_sizes)

        super(UpSamplingMultiDilationGenerator, self).__init__(
            latent_dim, window_size,
            [
                upsample_cls(4),
                MultiDilationBlock(latent_dim, n_filters, dilation_sizes),
                upsample_cls(4),
                MultiDilationBlock(n_filters * nd, n_filters, dilation_sizes),
                upsample_cls(4),
                MultiDilationBlock(n_filters * nd, n_filters, dilation_sizes),
                upsample_cls(4),
                MultiDilationBlock(n_filters * nd, n_filters, dilation_sizes),
                upsample_cls(4),
                MultiDilationBlock(n_filters * nd, n_filters, dilation_sizes),
                upsample_cls(4),
                MultiDilationBlock(n_filters * nd, n_filters, dilation_sizes),
                upsample_cls(2),
                MultiDilationBlock(n_filters * nd, 1, dilation_sizes,
                                   activation=lambda x: x),
                SumToSamples(),
                EnsureSize(window_size)
            ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class UpSamplingMultiResWaveNetGenerator(BaseGenerator):
    def __init__(
            self,
            latent_dim,
            window_size,
            n_filters,
            upsample_cls,
            kernel_sizes):
        nk = len(kernel_sizes)

        super(UpSamplingMultiResWaveNetGenerator, self).__init__(
            latent_dim, window_size, [
                upsample_cls(4),
                ConvBlock(latent_dim, n_filters, 3, 1, 1),
                upsample_cls(4),
                ConvBlock(n_filters, n_filters, 15, 1, 7),
                upsample_cls(4),
                MultiResolutionWaveNetBlock(n_filters, n_filters, kernel_sizes),
                upsample_cls(4),
                MultiResolutionWaveNetBlock(n_filters * nk, n_filters,
                                            kernel_sizes),
                upsample_cls(4),
                MultiResolutionWaveNetBlock(n_filters * nk, n_filters,
                                            kernel_sizes),
                upsample_cls(4),
                MultiResolutionWaveNetBlock(n_filters * nk, n_filters,
                                            kernel_sizes),
                upsample_cls(2),
                MultiResolutionWaveNetBlock(
                    n_filters * nk,
                    1,
                    kernel_sizes,
                    activation=lambda x: x),
                SumToSamples()
            ])
        self.kernel_sizes = kernel_sizes
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class DilatedGenerator(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters, upsample_cls):
        super(DilatedGenerator, self).__init__(latent_dim, window_size, [
            upsample_cls(4),
            DilatedBlock(latent_dim, n_filters, 1),
            upsample_cls(4),
            DilatedBlock(n_filters, n_filters, 2),
            upsample_cls(4),
            DilatedBlock(n_filters, n_filters, 4),
            upsample_cls(4),
            DilatedBlock(n_filters, n_filters, 6),
            upsample_cls(4),
            DilatedBlock(n_filters, n_filters, 8),
            upsample_cls(3),
            DilatedBlock(n_filters, n_filters, 8),
            upsample_cls(3),
            DilatedBlock(n_filters, 1, 8, activation=lambda x: x),
            SumToSamples(),
            EnsureSize(window_size)
        ])
        self.upsample_cls = upsample_cls

    def _extra_display_data(self):
        return dict(upsample=self.upsample_cls.__name__)


class NoiseTransformer(BaseGenerator):
    def __init__(self, latent_dim, window_size, n_filters):
        n_positional = 64
        in_channels = n_filters + latent_dim
        kernel_size = 16

        super(NoiseTransformer, self).__init__(latent_dim, window_size, [
            nn.Conv1d(1 + latent_dim + n_positional, n_filters, kernel_size, 1,
                      dilation=32, padding=16 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=16,
                      padding=8 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=8,
                      padding=4 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=4,
                      padding=2 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=2,
                      padding=1 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=1,
                      padding=1 * kernel_size),

            nn.Conv1d(in_channels, n_filters, kernel_size, 1,
                      dilation=32, padding=16 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=16,
                      padding=8 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=8,
                      padding=4 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=4,
                      padding=2 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=2,
                      padding=1 * kernel_size),
            nn.Conv1d(in_channels, n_filters, kernel_size, 1, dilation=1,
                      padding=1 * kernel_size),

            nn.Conv1d(in_channels, 1, kernel_size, 1, dilation=1,
                      padding=1 * kernel_size),
        ])

        pos = torch.cat(
            [torch.sin(torch.linspace(0, x * math.pi, 8192)) for x in
             torch.linspace(0.25, 4, n_positional)])
        pos = pos.view(n_positional, 8192)
        pos = pos.repeat(1, 1, 1)
        self.pos = Variable(pos, requires_grad=False).cuda()

        noise = torch.FloatTensor(1, 1, 8192)
        noise.normal_(0, 1)
        self.noise = Variable(noise, requires_grad=False).cuda()

    def forward(self, x):
        latent = x.view(-1, self.latent_dim, 1).repeat(1, 1, 8192)

        x = torch.cat([self.pos, self.noise], dim=1)

        for i, layer in enumerate(self.layers):
            z = torch.cat([x, latent], dim=1)
            x = layer(z)[..., :8192].contiguous()
            if i < len(self.layers) - 1:
                x = F.elu(x)

        return x


def test_generator(latent_dim, batch_size, generator):
    noise = torch.FloatTensor(3, latent_dim, 1)
    noise = Variable(noise)
    samples = generator(noise)
    print generator.__class__, samples.shape
    assert samples.shape == (batch_size, 1, 8192)


if __name__ == '__main__':
    latent_dim = 32
    batch_size = 3

    us_multiscale = MultiScaleUpSamplingGenerator(
        latent_dim, 8192, 16, LinearUpSamplingBlock)
    test_generator(latent_dim, batch_size, us_multiscale)

    multi_dilated = UpSamplingMultiDilationGenerator(
        latent_dim, 8192, 16, LinearUpSamplingBlock, [1, 3, 5, 7, 15, 31])
    test_generator(latent_dim, batch_size, multi_dilated)

    generator = ConvGenerator(latent_dim, 8192, 16)
    test_generator(latent_dim, batch_size, generator)

    mixed = PschyoAcousticUpsamplingGenerator(
        latent_dim, 8192, 16, LinearUpSamplingBlock)
    test_generator(latent_dim, batch_size, mixed)

    multiscale = MultiScaleGenerator(latent_dim, 8192, 8)
    test_generator(latent_dim, batch_size, multiscale)

    linear = UpSamplingGenerator(latent_dim, 8192, 16, LinearUpSamplingBlock)
    test_generator(latent_dim, batch_size, linear)

    nearest = UpSamplingGenerator(
        latent_dim, 8192, 16, NearestNeighborUpsamplingBlock)
    test_generator(latent_dim, batch_size, nearest)

    dct = UpSamplingGenerator(latent_dim, 8192, 16, DctUpSamplingBlock)
    test_generator(latent_dim, batch_size, dct)

    wavenet = UpSamplingWavenetGenerator(
        latent_dim, 8192, 16, LinearUpSamplingBlock)
    test_generator(latent_dim, batch_size, wavenet)

    multires = UpSamplingMultiResGenerator(
        latent_dim, 8192, 16, LinearUpSamplingBlock, [3, 9, 17, 25])
    test_generator(latent_dim, batch_size, multires)

    dilated = DilatedGenerator(latent_dim, 8192, 16, LinearUpSamplingBlock)
    test_generator(latent_dim, batch_size, dilated)

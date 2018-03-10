from torch import nn
from zounds.learn import DctTransform
from torch.nn import functional as F


class UpSamplingBlock(nn.Module):
    def __init__(self, mode, factor):
        super(UpSamplingBlock, self).__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        return F.upsample(x, scale_factor=self.factor, mode=self.mode)


class LinearUpSamplingBlock(UpSamplingBlock):
    upsampling_type = 'linear'

    def __init__(self, channels, factor):
        super(LinearUpSamplingBlock, self).__init__('linear', factor)


class NearestNeighborUpsamplingBlock(UpSamplingBlock):
    upsampling_type = 'nearest'

    def __init__(self, channels, factor):
        super(NearestNeighborUpsamplingBlock, self).__init__('nearest', factor)


class DctUpSamplingBlock(nn.Module):
    upsampling_type = 'dct'
    dct_transform = DctTransform(use_cuda=True)

    def __init__(self, channels, factor):
        super(DctUpSamplingBlock, self).__init__()
        self.factor = factor

    def forward(self, x):
        return self.dct_transform.dct_resample(x, self.factor, axis=-1)


class LearnedUpSamplingBlock(nn.Module):
    upsampling_type = 'learned'

    def __init__(self, channels, factor):
        super(LearnedUpSamplingBlock, self).__init__()
        self.upsample = nn.ConvTranspose1d(channels, channels, factor, factor)

    def forward(self, x):
        return self.upsample(x)

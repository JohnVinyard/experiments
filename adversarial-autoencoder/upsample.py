from torch import nn
from zounds.learn import DctTransform
from torch.nn import functional as F

"""
Upsampling Blocks
=================
learned
linear
nearest
dct
"""


class UpSamplingBlock(nn.Module):
    def __init__(self, mode, factor):
        super(UpSamplingBlock, self).__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        return F.upsample(
            x, scale_factor=self.factor, mode=self.mode)


class LinearUpSamplingBlock(UpSamplingBlock):

    def __init__(self, factor):
        super(LinearUpSamplingBlock, self).__init__('linear', factor)


class NearestNeighborUpsamplingBlock(UpSamplingBlock):

    def __init__(self, factor):
        super(NearestNeighborUpsamplingBlock, self).__init__('nearest', factor)


class DctUpSamplingBlock(nn.Module):
    dct_transform = DctTransform(use_cuda=True)

    def __init__(self, factor):
        super(DctUpSamplingBlock, self).__init__()
        self.factor = factor

    def cuda(self, device=None):
        self.dct_transform = self.dct_transform.cuda()
        return super(DctUpSamplingBlock, self).cuda(device=device)

    def forward(self, x):
        return self.dct_transform.dct_resample(x, self.factor, axis=-1)


class LearnedUpSamplingBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            activation=lambda x: x):

        super(LearnedUpSamplingBlock, self).__init__()
        self.activation = activation
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel, stride, padding)

    def forward(self, x):
        return self.activation(self.upsample(x))

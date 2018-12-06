import torch
from torch import nn
from torch.nn import functional as F


class SumToSamples(nn.Module):
    def __init__(self):
        super(SumToSamples, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=1, keepdim=True)


class CategoricalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(CategoricalBlock, self).__init__()
        self.layer = nn.Conv1d(
            in_channels, out_channels, kernel, stride, padding)

    def forward(self, x):
        return F.log_softmax(self.layer(x))


class EnsureSize(nn.Module):
    def __init__(self, desired_size):
        super(EnsureSize, self).__init__()
        self.desired_size = desired_size

    def forward(self, x):
        if x.shape[-1] == self.desired_size:
            return x

        if x.shape[-1] < self.desired_size:
            return F.pad(x, self.desired_size - x.shape[-1])
        else:
            return x[..., :self.desired_size].contiguous()

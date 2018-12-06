from torch.nn import functional as F
import torch
from zounds.learn import DctTransform
from torch import nn
from torch.optim import Adam
import numpy as np


class FeatureSpaceLoss(object):
    def __init__(self, network):
        super(FeatureSpaceLoss, self).__init__()
        self.network = network

    def cuda(self, device=None):
        self.network = self.network.cuda(device=device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        # then, try to minimize the distance in the network's feature space
        ti = self.network(input, return_features=True)
        tt = self.network(target, return_features=True)

        # error = -F.cosine_similarity(ti, tt).mean()

        error = ((ti - tt) ** 2).mean()
        return error


class LearnedLoss(object):
    def __init__(self, network, loss):
        super(LearnedLoss, self).__init__()
        self.network = network
        self.optim = Adam(
            network.parameters(),
            lr=0.0001,
            betas=(0, 0.9))
        self.loss = loss

    def cuda(self, device=None):
        self.network = self.network.cuda(device=device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        target = target.view(*input.shape)

        # first, train the network to assign the same inputs a zero output
        # next, train the network to assign the generated inputs a one output
        same = self.network(torch.cat([input, input], dim=1)).mean()
        diff = self.network(torch.cat([input, target], dim=1)).mean()
        error = same + -diff
        error.backward(retain_graph=True)
        self.optim.step()

        err = self.network(torch.cat([input, target], dim=1)).mean()

        return err  # + self.loss(input, target)


class CompositeLoss(object):
    def __init__(self, *losses):
        super(CompositeLoss, self).__init__()
        self.losses = losses

    def cuda(self, device=None):
        self.losses = [l.cuda() for l in self.losses]
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        target = target.view(input.shape)
        return sum(l(input, target) for l in self.losses)


class BandLoss2(nn.MSELoss):
    def __init__(self, factors):
        super(BandLoss2, self).__init__()
        self.factors = factors
        self.dct_transform = DctTransform()

    def cuda(self, device=None):
        self.dct_transform = DctTransform(use_cuda=True)
        return super(BandLoss2, self).cuda(device=device)

    def _transform(self, x):
        if isinstance(x, list):
            bands = x
        else:
            bands = self.dct_transform.frequency_decomposition(
                x, self.factors, axis=-1)
            bands = [b.squeeze() for b in bands]

        window = bands[0].shape[-1]
        stride = window

        b = []

        for band in bands:
            windowed = band.unfold(-1, window, stride) \
                .contiguous().view(-1, window)
            b.append(windowed)

        bands = torch.cat(b, dim=0)

        return bands

    def forward(self, input, target):
        input_bands = self._transform(input)
        target_bands = self._transform(target)

        spectral_loss = -F.cosine_similarity(
            input_bands, target_bands, dim=-1).mean()

        # input_norm = torch.norm(input_bands, dim=-1)
        # target_norm = torch.norm(target_bands, dim=-1)
        # print input_norm.shape, target_norm.shape
        #
        # norm_loss = -F.cosine_similarity(input_norm, target_norm, dim=-1).mean()

        return spectral_loss  # + norm_loss


class BandLoss3(nn.MSELoss):
    def __init__(self, factors):
        super(BandLoss3, self).__init__()
        self.factors = factors
        self.dct_transform = DctTransform()

    def cuda(self, device=None):
        self.dct_transform = DctTransform(use_cuda=True)
        return super(BandLoss3, self).cuda(device=device)

    def _transform(self, x):
        bands = self.dct_transform.frequency_decomposition(
            x, self.factors, axis=-1)
        return bands

    def _distance(self, x, y):
        # return -F.cosine_similarity(x, y, dim=-1).mean()
        x = torch.sign(x) * torch.log(torch.abs(x * 100) + 1)
        y = torch.sign(y) * torch.log(torch.abs(y * 100) + 1)
        return ((x - y) ** 2).mean()

    def forward(self, input, target):

        if isinstance(input, list):
            input_bands = input
        else:
            target = target.view(input.shape)
            input_bands = self._transform(input)

        target_bands = self._transform(target)

        target_bands = \
            [t.view(*i.shape) for t, i in zip(target_bands, input_bands)]

        spectral_loss = None
        for i, inp, t in zip(xrange(len(input_bands)), input_bands,
                             target_bands):
            if spectral_loss is None:
                spectral_loss = self._distance(inp, t)
            else:
                spectral_loss = spectral_loss + self._distance(inp, t)

        return spectral_loss


class FFTLoss(object):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def cuda(self, device=None):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        ti = torch.rfft(input.squeeze(), 1, normalized=True)
        tt = torch.rfft(target.squeeze(), 1, normalized=True)
        return -F.cosine_similarity(ti, tt).mean()
        return error

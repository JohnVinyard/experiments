from torch.optim import Adam
from torch.nn import functional as F
import torch
from torch import nn
from zounds.learn import DctTransform
from torch.nn.init import calculate_gain, orthogonal, xavier_normal


class LearnedPatchLoss(object):
    def __init__(self, network):
        super(LearnedPatchLoss, self).__init__()
        self.network = network
        for p in network.parameters():
            if p.data.dim() >= 2:
                p.data = xavier_normal(
                    p.data, gain=calculate_gain('leaky_relu', param=0.2))

            # if p.data.dim() == 3:
            #     x = p.data.normal_(0, 2. / p.data.shape[1])
            #     x = x.view(p.data.shape[0], -1)
            #     print x.shape
            #     u, s, v, = torch.svd(x)
            #     print p.shape, v.shape
            #     p.data = v.view(*p.data.shape)
            # elif p.data.dim() == 2:
            #     p.data.normal_(0, 1. / p.data.shape[1])
            # else:
            #     p.data.fill_(0)
            #     # self.optimizer = Adam(
            #     #     self.network.parameters(), lr=0.0001, betas=(0, 0.9))

    def cuda(self, device=None):
        self.network = self.network.cuda(device=device)
        return self

    # def _error(self, a, b):
    #     return F.cosine_similarity(a, b).mean()

    def forward(self, input, target):

        # if not input.volatile:
        #     self.network.zero_grad()
        #     for p in self.network.parameters():
        #         p.requires_grad = True
        #
        #     # first, teach the network to push input and target far apart
        #     ti = self.network(input).view(input.shape[0], -1)
        #     tt = self.network(target).view(input.shape[0], -1)
        #
        #     network_error = self._error(ti, tt)
        #     print 'Loss Error', network_error.data[0]
        #     network_error.backward(retain_graph=True)
        #     self.optimizer.step()

        # self.network.zero_grad()
        # then, try to minimize the distance in the network's feature space
        ti = self.network(input, return_features=True)
        tt = self.network(target, return_features=True)

        # mean_loss = torch.abs(input.mean() - target.mean())
        # std_loss = torch.abs(input.std() - target.std())
        # mse_loss = ((input - target) ** 2).mean()

        error = -F.cosine_similarity(ti, tt).mean()

        return error

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BandLoss2(nn.MSELoss):
    def __init__(self, factors):
        super(BandLoss2, self).__init__()
        self.factors = factors
        self.dct_transform = DctTransform()

    def cuda(self, device=None):
        self.dct_transform = DctTransform(use_cuda=True)
        return super(BandLoss2, self).cuda(device=device)

    def _transform(self, x):
        bands = self.dct_transform.frequency_decomposition(
            x, self.factors, axis=-1)

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
        target = target.view(input.shape)
        input_bands = self._transform(input)
        target_bands = self._transform(target)

        spectral_loss = -F.cosine_similarity(
            input_bands, target_bands, dim=-1).mean()

        mse_loss = super(BandLoss2, self).forward(input, target)

        return spectral_loss + (0.01 * mse_loss)


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
        return -F.cosine_similarity(x, y, dim=-1).mean()

        # x_norms = torch.norm(x, dim=-1, keepdim=True)
        # y_norms = torch.norm(y, dim=-1, keepdim=True)
        # x = x / (x_norms + 1e-8)
        # y = y / (y_norms + 1e-8)
        #
        # return super(BandLoss3, self).forward(x, y)

    def forward(self, input, target):
        target = target.view(input.shape)
        input_bands = self._transform(input)
        target_bands = self._transform(target)

        spectral_loss = None
        for inp, t in zip(input_bands, target_bands):
            if spectral_loss is None:
                spectral_loss = self._distance(inp, t)
            else:
                spectral_loss = spectral_loss + self._distance(inp, t)

        return spectral_loss + (
            super(BandLoss3, self).forward(input, target) * 0.01)

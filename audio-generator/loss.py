from torch import nn
from zounds.learn import DctTransform
import torch
from torch.nn import functional as F


class BandLoss2(nn.MSELoss):
    def __init__(self, factors):
        super(BandLoss2, self).__init__()
        self.factors = factors
        self.dct_transform = DctTransform()

    def cuda(self, device=None):
        self.dct_transform = DctTransform(use_cuda=True)
        return super(BandLoss2, self).cuda(device=device)

    def _transform(self, x):
        batch_size = x.shape[0]

        bands = self.dct_transform.frequency_decomposition(
            x, self.factors, axis=-1)

        window = 16
        stride = window

        b = []
        norms = []

        for band in bands:
            windowed = band.unfold(-1, window, stride) \
                .contiguous().view(batch_size, -1, window)
            normed = torch.norm(windowed, dim=-1)
            b.append(windowed.view(-1, window))
            norms.append(normed)

        bands = torch.cat(b, dim=0)
        coarse = torch.cat(norms, dim=-1)

        # bands = torch.cat([
        #     b.unfold(-1, window, stride).contiguous().view(-1, window)
        #     for b in bands
        # ], dim=0)

        # coarse = torch.norm(bands, dim=-1).view(x.shape[0], -1)
        return bands, coarse

    def forward(self, input, target):
        target = target.view(input.shape)
        input_bands, input_coarse = self._transform(input)
        target_bands, target_coarse = self._transform(target)

        coarse_loss = -F.cosine_similarity(
            input_coarse, target_coarse, dim=-1).mean()

        spectral_loss = -F.cosine_similarity(
            input_bands, target_bands, dim=-1).mean()

        return coarse_loss + spectral_loss

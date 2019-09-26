from __future__ import division
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, calculate_gain
from zounds.learn.util import \
    batchwise_unit_norm, batchwise_mean_std_normalization
import zounds


class EmbeddingNetwork(nn.Module):
    """
    Compute Log-scaled mel spectrogram, followed by a vanilla 2d convolutional
    network with alternating convolutional and max pooling layers
    """

    def __init__(self):
        super(EmbeddingNetwork, self).__init__()

        frequency_channels = 128
        channels = frequency_channels

        sr = zounds.SR11025()
        interval = zounds.FrequencyBand.audible_range(sr)
        scale = zounds.MelScale(interval, frequency_channels)
        self.bank = zounds.learn.FilterBank(
            samplerate=sr,
            kernel_size=512,
            scale=scale,
            scaling_factors=np.linspace(0.1, 1.0, len(scale)),
            normalize_filters=True,
            a_weighting=True)

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

        self.final = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 32, bias=False),
            nn.Linear(32, 16, bias=False),
            nn.Linear(16, 8, bias=False),
        )

        self.linear = nn.Linear(8, 3, bias=False)

    @classmethod
    def load_network(cls, weights_file_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = cls().to(device)

        try:
            # load network weights from a file on disk
            state_dict = torch.load(weights_file_path)
            network.load_state_dict(state_dict)
        except IOError:
            # There were no weights stored on disk.  Initialize them
            network.initialize_weights()

        return network, device

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

        for m in self.final.parameters():
            if m.data.dim() > 2:
                xavier_normal_(m.data, calculate_gain('leaky_relu', 0.2))

        for m in self.linear.parameters():
            if m.data.dim() > 2:
                xavier_normal_(m.data, 1)

    def forward(self, x):
        # normalize
        x = batchwise_mean_std_normalization(x)

        # filter bank
        x = self.bank(x, normalize=False)

        # temporal pooling
        x = F.avg_pool1d(x, 128, 64, padding=64)

        # give zero mean and unit variance
        x = zounds.learn.batchwise_mean_std_normalization(x)

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

        x = x.view(-1, self.final[0].in_features)
        for f in self.final:
            x = f(x)
            x = F.leaky_relu(x, 0.2)

        x = self.linear(x)
        x = batchwise_unit_norm(x)
        return x


import zounds
import argparse

from zounds.learn import GanExperiment, DctTransform
from torch import nn
from torch.nn import functional as F

import torch

dct_basis_cache = dict()
sample_size = 8192
scales = [0.5, 0.25, 0.125, 0.0625, 0.03125]
sizes = [int(sample_size * scale) for scale in scales]
latent_dim = 100
use_cuda = True

dct_transform = DctTransform(use_cuda=use_cuda)


def pytorch_frequency_decomposition(x, factors, axis=-1):
    bands = []
    factors = sorted(factors)
    for f in factors:
        rs = dct_transform.dct_resample(x, f, axis)
        bands.append(rs)
        us = dct_transform.dct_resample(rs, (1. / f), axis)
        x = x - us
    return bands


class Critic2(nn.Module):
    def __init__(self):
        super(Critic2, self).__init__()
        self.activation = F.elu
        self.dct_transform = DctTransform(use_cuda=use_cuda)

        self.l1 = nn.Conv1d(1, 64, 3, 2, 1, bias=False)

        self.l2_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l2 = nn.Conv1d(128, 64, 3, 2, 1, bias=False)

        self.l3_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l3 = nn.Conv1d(128, 64, 3, 2, 1, bias=False)

        self.l4_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l4 = nn.Conv1d(128, 64, 3, 2, 1, bias=False)

        self.l5_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l5 = nn.Conv1d(128, 64, 3, 2, 1, bias=False)

        self.main = nn.Sequential(
            nn.Conv1d(64, 128, 4, 2, 1, bias=False),
            nn.Conv1d(128, 128, 4, 2, 1, bias=False),
            nn.Conv1d(128, 256, 4, 2, 1, bias=False),
            nn.Conv1d(256, 512, 4, 2, 1, bias=False),
            nn.Conv1d(512, 512, 4, 2, 1, bias=False),
            nn.Conv1d(512, 512, 4, 4, 0, bias=False)
        )

        self.linear = nn.Linear(512, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, sample_size)
        bands = pytorch_frequency_decomposition(x, scales, axis=-1)

        bands_256 = bands[0]
        bands_512 = bands[1]
        bands_1024 = bands[2]
        bands_2048 = bands[3]
        bands_4096 = bands[4]

        features_4096 = self.l1(bands_4096)
        features_4096 = self.activation(features_4096)

        features_2048 = self.l2_audio(bands_2048)
        features_2048 = self.activation(features_2048)
        features_2048 = torch.cat([features_4096, features_2048], dim=1)
        features_2048 = self.l2(features_2048)
        features_2048 = self.activation(features_2048)

        features_1024 = self.l3_audio(bands_1024)
        features_1024 = self.activation(features_1024)
        features_1024 = torch.cat([features_2048, features_1024], dim=1)
        features_1024 = self.l3(features_1024)
        features_1024 = self.activation(features_1024)

        features_512 = self.l3_audio(bands_512)
        features_512 = self.activation(features_512)
        features_512 = torch.cat([features_1024, features_512], dim=1)
        features_512 = self.l3(features_512)
        features_512 = self.activation(features_512)

        features_256 = self.l3_audio(bands_256)
        features_256 = self.activation(features_256)
        features_256 = torch.cat([features_512, features_256], dim=1)
        features_256 = self.l3(features_256)
        features_256 = self.activation(features_256)

        x = features_256
        for m in self.main:
            x = m(x)
            x = self.activation(x)

        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        self.dct_transform = DctTransform(use_cuda=use_cuda)

        self.activation = F.elu

        # latent_dim => 256
        self.l1 = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.ConvTranspose1d(512, 256, 6, 4, 1, bias=False),
            nn.ConvTranspose1d(256, 128, 6, 4, 1, bias=False),
            nn.ConvTranspose1d(128, 64, 6, 4, 1, bias=False))
        self.l1_to_samples = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

        # 256 => 512
        self.l2_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l2 = nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False)
        self.l2_to_samples = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

        # 512 => 1024
        self.l3_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l3 = nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False)
        self.l3_to_samples = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

        # 1024 => 2048
        self.l4_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l4 = nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False)
        self.l4_to_samples = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

        # 2048 => 4096
        self.l5_audio = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.l5 = nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False)
        self.l5_to_samples = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

    def _apply_layer(
            self,
            samples,
            sample_layer,
            features,
            feature_layer,
            to_samples_layer):
        x = sample_layer(samples)
        x = self.activation(x)

        y = feature_layer(torch.cat([x, features], dim=1))
        output_features = self.activation(y)

        samples = to_samples_layer(y)
        return samples, output_features

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)

        for m in self.l1:
            x = m(x)
            x = self.activation(x)

        features_256 = x
        samples_256 = self.l1_to_samples(features_256)
        output = self.dct_transform.dct_resample(samples_256, 32)

        samples_512, features_512 = self._apply_layer(
            samples_256,
            self.l2_audio,
            features_256,
            self.l2,
            self.l2_to_samples)
        output = output + self.dct_transform.dct_resample(samples_512, 16)

        samples_1024, features_1024 = self._apply_layer(
            samples_512,
            self.l3_audio,
            features_512,
            self.l3,
            self.l3_to_samples)
        output = output + self.dct_transform.dct_resample(samples_1024, 8)

        samples_2048, features_2048 = self._apply_layer(
            samples_1024,
            self.l4_audio,
            features_1024,
            self.l4,
            self.l4_to_samples)
        output = output + self.dct_transform.dct_resample(samples_2048, 4)

        samples_4096, features_4096 = self._apply_layer(
            samples_2048,
            self.l5_audio,
            features_2048,
            self.l5,
            self.l5_to_samples)
        output = output + self.dct_transform.dct_resample(samples_4096, 2)
        return output


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = Generator2()
        self.discriminator = Critic2()

    def forward(self, x):
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--object-storage-region',
        help='the rackspace object storage region',
        default='DFW')
    parser.add_argument(
        '--object-storage-username',
        help='rackspace cloud username',
        required=True)
    parser.add_argument(
        '--object-storage-api-key',
        help='rackspace cloud api key',
        required=True)
    parser.add_argument(
        '--app-secret',
        help='app password',
        required=False)
    args = parser.parse_args()

    experiment = GanExperiment(
        'multiscale',
        zounds.InternetArchive('AOC11B'),
        GanPair(),
        latent_dim=latent_dim,
        sample_size=sample_size,
        sample_hop=256,
        n_critic_iterations=5,
        n_samples=int(1e5),
        app_port=8888,
        app_secret=args.app_secret,
        object_storage_username=args.object_storage_username,
        object_storage_api_key=args.object_storage_api_key,
        object_storage_region=args.object_storage_region)
    experiment.run()

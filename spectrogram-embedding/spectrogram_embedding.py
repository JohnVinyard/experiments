"""
Learn an embedding of short (~1.5 second) spectrograms, using the approach
described in

Unsupervised Learning of Semantic Audio Representations
https://arxiv.org/abs/1711.02209
"""

from __future__ import division
import argparse
from random import choice

import featureflow as ff
import numpy as np
import zounds
from scipy.signal import resample
from torch import nn
from torch.nn import functional as F
from zounds.learn import try_network
from zounds.spectral import apply_scale
from multiprocessing.pool import Pool, cpu_count

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

scale_bands = 96
embedding_dimension = 128
spectrogram_duration = 64

anchor_slice = slice(spectrogram_duration, spectrogram_duration * 2)

BasePipeline = zounds.learning_pipeline()


@zounds.simple_settings
class EmbeddingPipeline(BasePipeline):
    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=BasePipeline.shuffled)

    embedding = ff.PickleFeature(
        zounds.PyTorchNetwork,
        trainer=ff.Var('trainer'),
        post_training_func=(lambda x: x[:, anchor_slice].astype(np.float32)),
        needs=dict(data=scaled))

    unitnorm = ff.PickleFeature(
        zounds.UnitNorm,
        needs=embedding)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, embedding, unitnorm),
        store=True)


scale = zounds.GeometricScale(
    start_center_hz=50,
    stop_center_hz=samplerate.nyquist,
    bandwidth_ratio=0.115,
    n_bands=scale_bands)
scale.ensure_overlap_ratio()


def spectrogram(x):
    x = apply_scale(
        np.abs(x.real), scale, window=zounds.OggVorbisWindowingFunc())
    x = zounds.log_modulus(x * 100)
    x = x * zounds.AWeighting()
    return x.astype(np.float16)


windowing_scheme = zounds.HalfLapped()
spectrogram_sample_rate = zounds.SampleRate(
    frequency=windowing_scheme.frequency * (spectrogram_duration // 2),
    duration=windowing_scheme.frequency * spectrogram_duration)


@zounds.simple_lmdb_settings('sounds', map_size=1e10, user_supplied_id=True)
class Sound(BaseModel):
    short_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=windowing_scheme,
        wfunc=zounds.OggVorbisWindowingFunc(),
        needs=BaseModel.resampled)

    fft = zounds.ArrayWithUnitsFeature(
        zounds.FFT,
        padding_samples=1024,
        needs=short_windowed)

    geom = zounds.ArrayWithUnitsFeature(
        spectrogram,
        needs=fft)

    # a wider/longer slice of the spectrogram, purely computed here for the
    # benefit of the learning algorithm, since one way to choose positive
    # examples is to choose the spectrogram slice immediately preceding or
    # following the positive example
    log_spectrogram = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=windowing_scheme.frequency * (spectrogram_duration // 2),
            duration=windowing_scheme.frequency * spectrogram_duration * 3),
        needs=geom)

    ls = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=spectrogram_sample_rate,
        needs=geom)

    embedding = zounds.ArrayWithUnitsFeature(
        zounds.Learned,
        learned=EmbeddingPipeline(),
        dtype=np.float32,
        needs=ls)


def additive_noise(anchor, neighborhood):
    amt = np.random.uniform(0.01, 0.05)
    return anchor + np.random.normal(0, amt, anchor.shape).astype(anchor.dtype)


def nearby(anchor, neighborhood):
    slce = choice([
        slice(0, spectrogram_duration),
        slice(spectrogram_duration * 2, spectrogram_duration * 3)])
    return neighborhood[:, slce]


def time_stretch(anchor, neighborhood):
    factor = np.random.uniform(0.5, 1.5)
    new_size = int(factor * anchor.shape[1])
    rs = resample(anchor, new_size, axis=1).astype(anchor.dtype)
    if new_size > spectrogram_duration:
        return rs[:, :spectrogram_duration, :]
    else:
        diff = spectrogram_duration - new_size
        return np.pad(
            rs, ((0, 0), (0, diff), (0, 0)), mode='constant', constant_values=0)


def pitch_shift(anchor, neighborhood):
    amt = np.random.randint(-10, 10)
    shifted = np.roll(anchor, amt, axis=-1)
    if amt > 0:
        shifted[..., :amt] = 0
    else:
        shifted[..., amt:] = 0
    return shifted


class BasicConvolutionalBlock(nn.Module):
    """
    Two-dimensional convolution, batch norm, and a leaky ReLU applied in
    sequence
    """
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(BasicConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ResidualBlock(nn.Module):
    """
    A block that computes the residual after two sequential convolutions
    """
    def __init__(self, channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = BasicConvolutionalBlock(
            channels, channels, kernel, stride, padding)
        self.conv2 = BasicConvolutionalBlock(
            channels, channels, kernel, stride, padding)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.leaky_relu(x + residual, 0.2)


class IncreaseFeatureMapSizeBlock(nn.Module):
    """
    A block that increases the number of channels, without changing the other
    dimensions of the feature map
    """
    def __init__(self, in_channels, out_channels):
        super(IncreaseFeatureMapSizeBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, (3, 3), (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ResidualNetworkWithPooling(nn.Module):
    def __init__(
            self,
            spectrogram_duration=spectrogram_duration,
            scale_bands=scale_bands):
        super(ResidualNetworkWithPooling, self).__init__()
        self.spectrogram_duration = spectrogram_duration
        self.scale_bands = scale_bands

        self.main = nn.Sequential(
            IncreaseFeatureMapSizeBlock(1, 32),
            ResidualBlock(32),
            nn.MaxPool2d((3, 3), (2, 2)),

            IncreaseFeatureMapSizeBlock(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d((3, 3), (2, 2)),

            IncreaseFeatureMapSizeBlock(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d((3, 3), (2, 2)),

            IncreaseFeatureMapSizeBlock(128, 256),
            ResidualBlock(256),
            nn.MaxPool2d((3, 3), (2, 2)),

            BasicConvolutionalBlock(256, 512, (1, 3), (1, 3), (0, 0))
        )

        self.l1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256, embedding_dimension, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, self.spectrogram_duration, self.scale_bands)
        for m in self.main:
            x = m(x)
        x = x.view(-1, 512)
        x = self.l1(x)
        x = self.bn1(x)
        x = self.l2(x)
        return x


def access_log_spectrogram(snd):
    print snd._id
    x = snd.log_spectrogram
    del snd
    return x


def test_network():
    network = ResidualNetworkWithPooling(spectrogram_duration, scale_bands)
    x = np.random.normal(0, 1, (32, 64, 96)).astype(np.float32)
    output = try_network(network, x)
    print output.size()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-network',
        help='test the network with random noise and exit',
        action='store_true')
    parser.add_argument(
        '--force-train',
        help='retrain the model even if it exists',
        action='store_true')
    parser.add_argument(
        '--epochs',
        help='how many passes over the data should be made during training',
        type=int)
    parser.add_argument(
        '--init-weights',
        help='should weights be loaded from the existing model',
        action='store_true')
    parser.add_argument(
        '--batch-size',
        help='how many examples constitute a minibatch?',
        type=int,
        default=64)
    parser.add_argument(
        '--repl',
        help='just start up a repl to interact with features',
        action='store_true')
    parser.add_argument(
        '--nsamples',
        help='the number of samples to draw from the database for training',
        type=int)
    parser.add_argument(
        '--search',
        help='build a brute force search index, and interact in-browser',
        action='store_true')

    args = parser.parse_args()

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.geom,
        globals=globals(),
        locals=locals())
    port = 8888

    if args.repl:
        app.start(port=port)
    elif args.test_network:
        test_network()
    elif args.search:
        ep = EmbeddingPipeline()

        def g():
            for snd in Sound:
                print snd._id
                yield snd._id, snd.embedding

        search = zounds.BruteForceSearch(g())
        app.start(port=port)
    elif not EmbeddingPipeline.exists() or args.force_train:
        network = ResidualNetworkWithPooling(spectrogram_duration, scale_bands)

        if args.init_weights:
            embedding = EmbeddingPipeline()
            weights = embedding.pipeline[1].network.state_dict()
            network.load_state_dict(weights)
            print 'initialized weights'

        trainer = zounds.TripletEmbeddingTrainer(
            network,
            epochs=args.epochs,
            batch_size=args.batch_size,
            anchor_slice=anchor_slice,
            deformations=[nearby, pitch_shift, time_stretch, additive_noise])

        pool = Pool(cpu_count())
        iterator = pool.imap_unordered(access_log_spectrogram, Sound)

        EmbeddingPipeline.process(
            samples=iterator,
            trainer=trainer,
            nsamples=args.nsamples,
            dtype=np.float16)

        pool.close()
        pool.join()
        app.start(port=port)

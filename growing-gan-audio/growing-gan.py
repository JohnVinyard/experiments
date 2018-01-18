"""
Attempt to use some of the tricks for incrementally training a generative
adversarial network at different scales from this paper:

Progressive Growing of GANs for Improved Quality, Stability, and Variation
https://arxiv.org/abs/1710.10196

I apply these tricks to a GAN in one dimension, in the audio domain.
"""

from __future__ import division
import featureflow as ff
import numpy as np
from torch import nn
from torch.nn import functional as F
from random import choice
import torch
from scipy.signal import resample

import zounds
from zounds.learn import to_var, from_var, try_network

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

latent = 128
window_size = 8192
sizes = [2 ** i for i in xrange(3, 14)]
epochs_per_layer = 20


@zounds.simple_lmdb_settings(
    'growing_gan', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=samplerate.frequency * 4096,
            duration=samplerate.frequency * window_size),
        needs=BaseModel.resampled)


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            upsample=True):
        super(GeneratorBlock, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1 if upsample else 0,
            bias=False)
        self.to_samples = nn.Conv1d(out_channels, 1, 1, bias=False)

    def sample_norm(self, x):
        """
        pixel norm as described in section 4.2 here:
        https://arxiv.org/pdf/1710.10196.pdf
        """
        original = x
        # square
        x = x ** 2
        # feature-map-wise sum
        x = torch.sum(x, dim=1)
        # scale by number of feature maps
        x *= 1.0 / original.shape[1]
        x += 10e-8
        x = torch.sqrt(x)
        return original / x.view(-1, 1, x.shape[-1])

    def forward(self, x):
        if self.upsample:
            x = F.upsample(x, scale_factor=2, mode='linear')

        x = self.conv(x)
        x = self.sample_norm(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.sample_norm(x)
        x = F.leaky_relu(x, 0.2)

        features = x
        samples = self.to_samples(features)
        return features, samples


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            GeneratorBlock(latent, 512, 8, 1, 8, upsample=False),
            GeneratorBlock(512, 256, 3, 1, 1, upsample=True),
            GeneratorBlock(256, 256, 3, 1, 1, upsample=True),
            GeneratorBlock(256, 128, 3, 1, 1, upsample=True),
            GeneratorBlock(128, 64, 3, 1, 1, upsample=True),
            GeneratorBlock(64, 64, 3, 1, 1, upsample=True),
            GeneratorBlock(64, 64, 3, 1, 1, upsample=True),
            GeneratorBlock(64, 32, 3, 1, 1, upsample=True),
            GeneratorBlock(32, 32, 3, 1, 1, upsample=True),
            GeneratorBlock(32, 32, 3, 1, 1, upsample=True),
            GeneratorBlock(32, 16, 3, 1, 1, upsample=True))

    def fade(self, low_res, high_res, alpha):
        up = F.upsample(low_res, scale_factor=2, mode='nearest')
        return (up * (1 - alpha)) + (high_res * alpha)

    def forward(self, z, phase):
        z = z.view(-1, latent, 1)

        layer = np.ceil(phase)
        alpha = phase % 1

        features = z
        samples = None

        for i in xrange(int(layer) + 1):
            features, new_samples = self.main[i](features)
            if alpha and i == layer and i > 0:
                samples = self.fade(samples, new_samples, alpha)
            else:
                samples = new_samples
        return samples


def downsample_audio(x, ratio):
    return x[:, :, ::ratio]


class DiscriminatorBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            downsample=True):
        super(DiscriminatorBlock, self).__init__()
        self.from_samples = nn.Conv1d(1, in_channels, 1, 1, bias=False)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1 if downsample else 0,
            bias=False)
        self.downsample = downsample

    def samples_to_features(self, samples):
        return self.from_samples(samples)

    def forward(self, x):
        original_samples = x

        if x.shape[1] == 1:
            x = self.samples_to_features(x)

        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        if self.downsample:
            x = F.avg_pool1d(x, 2, 2)

        features = x
        return features, original_samples


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear_size = 512
        self.main = nn.Sequential(
            DiscriminatorBlock(1, 16, 3, 1, 1, downsample=True),
            DiscriminatorBlock(16, 16, 3, 1, 1, downsample=True),
            DiscriminatorBlock(16, 32, 3, 1, 1, downsample=True),
            DiscriminatorBlock(32, 32, 3, 1, 1, downsample=True),
            DiscriminatorBlock(32, 64, 3, 1, 1, downsample=True),
            DiscriminatorBlock(64, 64, 3, 1, 1, downsample=True),
            DiscriminatorBlock(64, 128, 3, 1, 1, downsample=True),
            DiscriminatorBlock(128, 256, 3, 1, 1, downsample=True),
            DiscriminatorBlock(256, 256, 3, 1, 1, downsample=True),
            DiscriminatorBlock(256, 512, 3, 1, 1, downsample=True),
            DiscriminatorBlock(512, 512, 8, 1, 1, downsample=False),
        )
        self.linear = nn.Linear(self.linear_size, 1)

    def fade(self, input_features, input_samples, current_layer, alpha):
        down = downsample_audio(input_samples, 2)
        down_features = current_layer.samples_to_features(down)
        return (input_features * alpha) + (down_features * (1 - alpha))

    def forward(self, samples, phase):
        layer = np.ceil(phase)
        alpha = phase % 1

        input_samples = samples
        features = samples

        layers = list(self.main)
        layers = layers[-(int(layer) + 1):]

        for i, layer in enumerate(layers):
            if alpha and i == 1:
                features = self.fade(features, input_samples, layer, alpha)
            features, input_samples = layer(features)

        features = features.view(-1, self.linear_size)
        return self.linear(features)


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self):
        raise NotImplementedError()


BasePipeline = zounds.learning_pipeline(np.float32)


@zounds.simple_settings
class GrowingGanPipeline(BasePipeline):
    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=BasePipeline.shuffled)

    network = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=ff.Var('trainer'),
        needs=scaled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, network),
        store=True)


network = GanPair().cuda()


def test_network():
    phases = np.arange(0, len(sizes) - 1 + 0.5, 0.5)
    gan = GanPair().cuda()
    for phase in phases:
        x = to_var(np.random.normal(0, 1, (64, latent)).astype(np.float32))
        phase
        print phase, '========================================'
        fake = gan.generator.forward(x, phase)
        print 'generated', fake.shape
        score = gan.discriminator.forward(fake, phase)
        print 'scored', score.shape


def epoch_to_phase(epoch):
    """
    produce the current phase argument, given the current epoch
    """
    current = epoch / epochs_per_layer
    rounded = int(current)
    if rounded % 2:
        return (rounded // 2) + (current % 1)
    else:
        return rounded // 2


real_samples = [None]
generated_samples = [None]


def real_sample():
    return choice(real_samples[0]).squeeze()


def fake_sample():
    return choice(generated_samples[0]).squeeze()


def audio(raw):
    upsampled = resample(raw, window_size)
    return zounds \
        .AudioSamples(upsampled, samplerate) \
        .pad_with_silence(zounds.Seconds(1))


def real_audio():
    return audio(real_sample())


def fake_audio():
    return audio(fake_sample())


def fake_stft():
    from scipy.signal import stft
    return zounds.log_modulus(stft(fake_sample())[2].T * 100)


def preprocess(epoch, input_v):
    """
    Down-sample real examples for the critic, when necessary
    """
    input_v = input_v.view(-1, 1, input_v.shape[-1])

    # lookup the correct size for this epoch
    phase = epoch_to_phase(epoch)
    index = int(np.ceil(phase))
    size = sizes[index]
    if input_v.shape[-1] == size:
        return input_v
    else:
        ratio = int(input_v.shape[-1] // size)
        x = downsample_audio(input_v, ratio)
        real_samples[0] = from_var(x).squeeze()
        return x


def arg_maker(epoch):
    return dict(phase=epoch_to_phase(epoch))


def batch_complete(epoch, network):
    phase = epoch_to_phase(epoch)
    z = np.random.normal(0, 1, (100, latent)).astype(np.float32)
    generated = from_var(try_network(network.generator, z, phase=phase))
    generated_samples[0] = generated.squeeze()


if __name__ == '__main__':

    for p in network.parameters():
        p.data.normal_(0, 0.2)

    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())

    with app.start_in_thread(9999):
        zounds.ingest(
            zounds.InternetArchive('AOC11B'),
            Sound,
            multi_threaded=True)

        trainer = zounds.WassersteinGanTrainer(
            network,
            (latent,),
            n_critic_iterations=10,
            epochs=1000,
            batch_size=32,
            preprocess_minibatch=preprocess,
            kwargs_factory=arg_maker,
            on_batch_complete=batch_complete)


        def gen():
            """
            Return only a single sound, and exclude the last ten seconds, since
            that contains speech instead of piano
            """
            sounds = list(Sound)
            for sound in sounds:
                duration = sound.windowed.dimensions[0].end - zounds.Seconds(10)
                ts = zounds.TimeSlice(duration=duration)
                yield sound.windowed[ts]


        if not GrowingGanPipeline.exists():
            GrowingGanPipeline.process(
                samples=gen(),
                nsamples=int(5e5),
                trainer=trainer)

    app.start(9999)

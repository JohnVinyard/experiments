from __future__ import division
import zounds
import featureflow as ff
from zounds.learn import Conv1d, ConvTranspose1d, to_var, from_var
from torch import nn
from torch.nn import functional as F
import numpy as np
from random import choice
import torch

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

sample_size = 8192
latent_dim = 100


def activation(x):
    return F.leaky_relu(x, 0.2)


window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * sample_size,
    duration=samplerate.frequency * sample_size)


@zounds.simple_lmdb_settings('bach_wgan', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_sample_rate,
        needs=BaseModel.resampled)


def sample_norm(x):
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


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(UpSamplingBlock, self).__init__()
        self.upscale_factor = upscale_factor
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        x = F.upsample(x, scale_factor=self.upscale_factor, mode='linear')
        for m in self.main:
            x = m(x)
            x = sample_norm(x)
            x = activation(x)
        return x


class UpSamplingGenerator(nn.Module):
    def __init__(self):
        super(UpSamplingGenerator, self).__init__()
        self.main = nn.Sequential(
            UpSamplingBlock(latent_dim, 512, 8),
            UpSamplingBlock(512, 256, 4),
            UpSamplingBlock(256, 128, 4),
            UpSamplingBlock(128, 128, 4),
            UpSamplingBlock(128, 128, 4),
            UpSamplingBlock(128, 64, 4),
        )
        self.final = nn.Conv1d(64, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        x = self.main(x)
        x = self.final(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale_factor):
        super(DownsamplingBlock, self).__init__()
        self.downscale_factor = downscale_factor
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        if self.downscale_factor > 1:
            x = F.avg_pool1d(x, self.downscale_factor)
        for m in self.main:
            x = m(x)
            x = activation(x)
        return x


# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.main = nn.Sequential(
#             DownsamplingBlock(1, 8, 1),
#             DownsamplingBlock(8, 16, 4),
#             DownsamplingBlock(16, 32, 4),
#             DownsamplingBlock(32, 64, 4),
#             DownsamplingBlock(64, 128, 4),
#             DownsamplingBlock(128, 256, 4),
#             DownsamplingBlock(256, 512, 8),
#         )
#         self.linear = nn.Linear(512, 1, bias=False)
#
#     def forward(self, x):
#         x = x.view(-1, 1, sample_size)
#         x = self.main(x)
#         x = x.view(-1, 512)
#         x = self.linear(x)
#         return x


class Critic(nn.Module):
    def __init__(self, input_channels=1):
        super(Critic, self).__init__()
        self.input_channels = input_channels
        self.last_dim = 512
        self.main = nn.Sequential(
            Conv1d(
                input_channels, 128, 16, 8, 4,
                batch_norm=False, dropout=False, activation=activation),
            Conv1d(
                128, 128, 8, 4, 2,
                batch_norm=False, dropout=False, activation=activation),
            Conv1d(
                128, 128, 8, 4, 2,
                batch_norm=False, dropout=False, activation=activation),
            Conv1d(
                128, 256, 8, 4, 2,
                batch_norm=False, dropout=False, activation=activation),
            Conv1d(
                256, 256, 8, 4, 2,
                batch_norm=False, dropout=False, activation=activation),
            Conv1d(
                256, self.last_dim, 4, 1, 0,
                batch_norm=False, dropout=False, activation=activation))
        self.linear = nn.Linear(self.last_dim, 1)

    def forward(self, x):
        x = x.view(-1, self.input_channels, sample_size)
        x = self.main(x)
        x = x.view(-1, self.last_dim)
        x = self.linear(x)
        return x


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = UpSamplingGenerator()
        self.discriminator = Critic()

    def forward(self, x):
        raise NotImplementedError()


BasePipeline = zounds.learning_pipeline()


@zounds.simple_settings
class Gan(BasePipeline):
    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=BasePipeline.shuffled)

    wgan = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=ff.Var('trainer'),
        needs=scaled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, wgan,),
        store=True)


fake_samples = [None]


def batch_complete(epoch, network, samples):
    samples = from_var(samples).squeeze()
    fake_samples[0] = samples


def fake_audio():
    sample = choice(fake_samples[0])
    return zounds \
        .AudioSamples(sample, samplerate) \
        .pad_with_silence(zounds.Seconds(1))


def fake_stft():
    from scipy.signal import stft
    return zounds.log_modulus(stft(fake_audio())[2].T * 100)


if __name__ == '__main__':

    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)

    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())


    def gen():
        """
        Exclude the last 10 seconds of each sound, since we know that each
        Bach piece includes some human speech at the end
        """
        for sound in Sound:
            start = zounds.Seconds(10)
            end = sound.windowed.dimensions[0].end - zounds.Seconds(10)
            duration = end - start
            ts = zounds.TimeSlice(start=start, duration=duration)
            yield sound.windowed[ts]


    with app.start_in_thread(8888):
        if not Gan.exists():
            network = GanPair()

            for p in network.parameters():
                p.data.normal_(0, 0.02)

            trainer = zounds.WassersteinGanTrainer(
                network,
                latent_dimension=(latent_dim,),
                n_critic_iterations=5,
                epochs=500,
                batch_size=32,
                on_batch_complete=batch_complete)

            Gan.process(
                samples=gen(),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float32)

    app.start(8888)

from __future__ import division
import zounds
import featureflow as ff
from zounds.learn import Conv1d, ConvTranspose1d, to_var, from_var
from torch import nn
from torch.nn import functional as F
import numpy as np
from random import choice

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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ConvTranspose1d(
                latent_dim, 512, 4, 1, 0,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 256, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                256, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                128, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                128, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                128, 64, 16, 8, 4,
                sample_norm=True, dropout=False, activation=activation),
            Conv1d(64, 1, 1, 1, 0,
                   dropout=False,
                   batch_norm=False,
                   sample_norm=False,
                   activation=None)
        )

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)

        for m in self.main:
            nx = m(x)
            factor = nx.shape[1] // x.shape[1]
            if nx.shape[-1] == x.shape[-1] and factor:
                upsampled = F.upsample(x, scale_factor=factor, mode='linear')
                x = activation(x + upsampled)
            else:
                x = nx

        return x.view(-1, sample_size)


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
        self.generator = Generator()
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


def batch_complete(epoch, network):
    latents = np.random.normal(0, 1, (100, latent_dim)).astype(np.float32)
    latents = to_var(latents)
    samples = network.generator(latents)
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
                n_critic_iterations=15,
                epochs=500,
                batch_size=32,
                on_batch_complete=batch_complete)

            Gan.process(
                samples=gen(),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float32)

    app.start(8888)

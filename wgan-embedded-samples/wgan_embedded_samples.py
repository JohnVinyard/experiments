from __future__ import division

from random import choice

import featureflow as ff
import numpy as np
import torch
import zounds
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from zounds.learn import Conv1d, ConvTranspose1d
from zounds.learn.util import from_var

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

latent_dim = 100
sample_size = 8192
audio_embedding_dimension = 3

window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * 1024,
    duration=samplerate.frequency * sample_size)


@zounds.simple_lmdb_settings(
    'audio_embedding', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    long_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_sample_rate,
        needs=BaseModel.resampled)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.activation = lambda x: F.leaky_relu(x, 0.2)
        self.main = nn.Sequential(
            ConvTranspose1d(
                latent_dim, 512, 4, 1, 0,
                sample_norm=True, dropout=False, activation=self.activation),
            ConvTranspose1d(
                512, 256, 8, 4, 2,
                sample_norm=True, dropout=False, activation=self.activation),
            ConvTranspose1d(
                256, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=self.activation),
            ConvTranspose1d(
                128, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=self.activation),
            ConvTranspose1d(
                128, 128, 8, 4, 2,
                sample_norm=True, dropout=False, activation=self.activation),
            ConvTranspose1d(
                128, 1, 16, 8, 4,
                sample_norm=False,
                batch_norm=False,
                dropout=False,
                activation=None),
        )

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        for m in self.main:
            nx = m(x)
            factor = nx.shape[1] // x.shape[1]
            if nx.shape[-1] == x.shape[-1] and factor:
                upsampled = F.upsample(x, scale_factor=factor, mode='linear')
                x = self.activation(x + upsampled)
            else:
                x = nx
        return x


class RawSampleEmbedding(nn.Module):
    def __init__(self):
        super(RawSampleEmbedding, self).__init__()
        self.linear = nn.Linear(256, audio_embedding_dimension)

    def categorical(self, x):
        x = x.view(-1)

        # mu-law
        m = Variable(torch.FloatTensor(1), requires_grad=False).cuda()
        m[:] = 255 + 1

        s = torch.sign(x)
        x = torch.abs(x)
        x = s * (torch.log(1 + (255 * x)) / torch.log(m))

        # shift and scale from [-1, 1] to [0, 255]
        x = x + 1
        x = x * (256 / 2.)

        # categorical encoding
        y = Variable(torch.arange(0, 256), requires_grad=False).cuda()
        x = -(((x[..., None] - y) ** 2) * 1e2)
        x = F.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.categorical(x)

        # embed the categorical variables
        x = self.linear(x)

        # place all embeddings on the unit sphere
        norms = torch.norm(x, dim=-1)
        x = x / norms.view(-1, 1)
        x = x.view(-1, audio_embedding_dimension, sample_size)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.last_dim = 512
        self.embedding = RawSampleEmbedding()
        self.activation = F.elu
        self.main = nn.Sequential(
            Conv1d(
                audio_embedding_dimension, 128, 16, 8, 4,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                128, 128, 8, 4, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                128, 128, 8, 4, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                128, 256, 8, 4, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                256, 256, 8, 4, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                256, self.last_dim, 4, 1, 0,
                batch_norm=False, dropout=False, activation=self.activation))
        self.linear = nn.Linear(self.last_dim, 1, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, audio_embedding_dimension, sample_size)
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


BaseGanPipeline = zounds.learning_pipeline()


@zounds.simple_settings
class Gan(BaseGanPipeline):
    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=BaseGanPipeline.shuffled)

    wgan = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=ff.Var('trainer'),
        needs=scaled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, wgan),
        store=True)


fake_samples = [None]

if __name__ == '__main__':
    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)

    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.long_windowed,
        globals=globals(),
        locals=locals())

    # then, train a GAN to produce fake *embedded* samples
    with app.start_in_thread(8888):
        if not Gan.exists():
            gan_pair = GanPair()

            for p in gan_pair.parameters():
                p.data.normal_(0, 0.02)


            def batch_complete(epoch, gan_network, samples):
                fake_samples[0] = from_var(samples)


            def one_hot(x):
                # mu law encode
                x = zounds.mu_law(x, mu=255)
                # quantize
                x = (255 * ((x * 0.5) + 0.5))
                x = x.astype(np.int64)
                x = zounds.ArrayWithUnits(x, x.dimensions)
                return x


            def inverse_one_hot(x):
                x = x.astype(np.float32)
                x /= 255.
                x -= 0.5
                x *= 2
                x = zounds.inverse_mu_law(x)
                x = zounds.AudioSamples(x, samplerate)
                return x


            def fake_audio():
                sample = choice(fake_samples[0]).squeeze()
                sample = zounds.AudioSamples(sample, samplerate)
                sample = one_hot(sample)
                sample = inverse_one_hot(sample)
                return zounds \
                    .AudioSamples(sample, samplerate) \
                    .pad_with_silence(zounds.Seconds(1))


            def fake_stft():
                from zounds.spectral import stft, rainbowgram
                samples = fake_audio()
                wscheme = zounds.SampleRate(
                    frequency=samples.samplerate.frequency * 128,
                    duration=samples.samplerate.frequency * 256)
                coeffs = stft(samples, wscheme, zounds.HanningWindowingFunc())
                return rainbowgram(coeffs)


            trainer = zounds.WassersteinGanTrainer(
                gan_pair,
                latent_dimension=(latent_dim,),
                n_critic_iterations=5,
                epochs=1000,
                batch_size=6,
                on_batch_complete=batch_complete)

            Gan.process(
                samples=(snd.long_windowed for snd in Sound),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float16)

    app.start(8888)

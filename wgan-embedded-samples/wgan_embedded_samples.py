from __future__ import division
import zounds
import numpy as np
import featureflow as ff
from torch import nn
from torch.nn import functional as F
from torch import optim
from scipy.spatial.distance import cdist
from zounds.learn.util import to_var, from_var
from zounds.learn import Conv1d, ConvTranspose1d
from random import choice
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

latent_dim = 100
sample_size = 8192
audio_embedding_dimension = 3

window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * sample_size,
    duration=samplerate.frequency * sample_size)


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


@zounds.simple_lmdb_settings(
    'audio_embedding', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    oh = zounds.ArrayWithUnitsFeature(
        one_hot,
        needs=BaseModel.resampled)

    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=samplerate.frequency,
            duration=samplerate.frequency * 2),
        wfunc=None,
        needs=oh)

    long_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_sample_rate,
        needs=oh)


class UnitNormEmbedding(nn.Module):
    def __init__(self):
        super(UnitNormEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            256, audio_embedding_dimension)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        norms = torch.norm(x, dim=-1).view(-1, 1)
        x = x / norms
        return x


class AudioEmbedding(nn.Module):
    def __init__(self):
        super(AudioEmbedding, self).__init__()
        self.embedding = UnitNormEmbedding()
        self.linear = nn.Linear(audio_embedding_dimension, 256, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = F.log_softmax(x)
        return x


BaseEmbeddingPipeline = zounds.learning_pipeline()


@zounds.simple_settings
class AudioEmbeddingPipeline(BaseEmbeddingPipeline):
    network = ff.PickleFeature(
        zounds.PyTorchNetwork,
        trainer=ff.Var('trainer'),
        needs=BaseEmbeddingPipeline.samples,
        # TODO: Resolve inconsistent
        # MRO when using KeySelector and AspectExtractor
        training_set_prep=lambda data: data['samples'])

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(network,),
        store=True)


def activation(x):
    return F.leaky_relu(x, 0.2)


window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * sample_size,
    duration=samplerate.frequency * sample_size)


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
            Conv1d(64, 3, 1, 1, 0,
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

        x = x.view(-1, audio_embedding_dimension, sample_size)
        norms = torch.norm(x, dim=1)
        x = x / norms.unsqueeze(1)
        return x


class RawSampleEmbedding(nn.Module):
    def __init__(self):
        super(RawSampleEmbedding, self).__init__()
        self.linear = nn.Linear(256, 3)

    # TODO: can I make this fit into memory if some of the variables below are
    # defined as volatile/requires_grad=False?
    def forward(self, x):
        original_shape = x.shape

        # shift and scale from [-1, 1] to [0, 255]
        x = x.view(-1)
        x = x + 1
        x = x * (256 / 2.)

        # categorical encoding
        y = torch.arange(0, 256)
        # x = -torch.abs(y - x[..., None])
        x = -(((x[..., None] - y) ** 2) * 1e8)
        x = F.softmax(x)

        # embed the categorical variables
        x = self.linear(x)

        # place all embeddings on the unit spheroid
        norms = torch.norm(x, dim=-1)
        x = x / norms.view(-1, 1)
        x = x.view(-1, 3, original_shape[-1])
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.last_dim = 512
        self.main = nn.Sequential(
            Conv1d(
                audio_embedding_dimension, 128, 16, 8, 4,
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
    wgan = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=ff.Var('trainer'),
        needs=BaseGanPipeline.shuffled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(wgan,),
        store=True)


fake_samples = [None]

if __name__ == '__main__':
    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)


    # first, train a network to embed audio samples
    def gen():
        for snd in Sound:
            yield dict(data=snd.windowed[..., 0], labels=snd.windowed[..., 1])


    trainer = zounds.SupervisedTrainer(
        model=AudioEmbedding(),
        loss=nn.NLLLoss(),
        optimizer=lambda model:
        optim.Adam(model.parameters(), lr=0.00005),
        epochs=50,
        batch_size=256,
        holdout_percent=0.2)

    if not AudioEmbeddingPipeline.exists():
        AudioEmbeddingPipeline.process(
            samples=gen(),
            trainer=trainer,
            nsamples=int(1e7),
            dtype=np.int64)

    aep = AudioEmbeddingPipeline()
    network = aep.pipeline[0].network
    embedding_weights = network.embedding.embedding.weight.cpu().data.numpy()


    def plot_embedding():
        q = torch.arange(0, 255).long()
        q = Variable(q).cuda()
        embedded = network.embedding(q)
        points = from_var(embedded)
        print points.shape

        fig = plt.figure()
        dim = points.shape[-1]
        ax = fig.add_subplot(111, projection='3d')
        coords = [points[:, i] for i in xrange(dim)]
        x = ax.scatter(*coords)
        fig = x.get_figure()
        fig.set_size_inches((50, 50))
        plt.savefig('embedding')


    plot_embedding()

    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())

    # then, train a GAN to produce fake *embedded* samples
    with app.start_in_thread(8888):
        if not Gan.exists():
            gan_pair = GanPair()

            for p in gan_pair.parameters():
                p.data.normal_(0, 0.02)


            # TODO: try distance on the unit ball/sphere instead
            def unembed_samples(x):
                """
                transform embedded samples to raw audio samples
                """
                dist = cdist(embedding_weights, x, metric='cosine')
                indices = np.argmin(dist, axis=0)
                return inverse_one_hot(indices)


            def batch_complete(epoch, gan_network, samples):
                fake_samples[0] = from_var(samples)


            def unembed(samples):
                samples = samples.reshape(
                    (-1, audio_embedding_dimension))
                raw_samples = unembed_samples(samples)
                raw_samples = raw_samples.reshape((-1, sample_size))
                return raw_samples[0]


            def fake_audio():
                sample = choice(fake_samples[0])
                sample = unembed(sample)
                return zounds \
                    .AudioSamples(sample, samplerate) \
                    .pad_with_silence(zounds.Seconds(1))


            def fake_stft():
                from scipy.signal import stft
                return zounds.log_modulus(stft(fake_audio())[2].T * 100)


            trainer = zounds.WassersteinGanTrainer(
                gan_pair,
                latent_dimension=(latent_dim,),
                n_critic_iterations=10,
                epochs=1000,
                batch_size=32,
                on_batch_complete=batch_complete)


            def embedded_samples(x):
                original = x
                x = to_var(x).view(1, -1)
                embedded = network.embedding(x)
                embedded = from_var(embedded).reshape(
                    original.shape + (audio_embedding_dimension,))
                return zounds.ArrayWithUnits(
                    embedded,
                    original.dimensions + (zounds.IdentityDimension(),))


            Gan.process(
                samples=(embedded_samples(snd.long_windowed) for snd in Sound),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float16)

    app.start(8888)

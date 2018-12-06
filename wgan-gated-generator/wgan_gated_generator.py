from __future__ import division
import zounds
import featureflow as ff
from zounds.learn import from_var, sample_norm, apply_network
from torch import nn
from torch.nn import functional as F
import numpy as np
from random import choice
import torch

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

sample_size = 8192
latent_dim = 100


window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * 1024,
    duration=samplerate.frequency * sample_size)


@zounds.simple_lmdb_settings('gated_wgan', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_sample_rate,
        needs=BaseModel.resampled)


class LayerWithAttention(nn.Module):
    def __init__(
            self,
            layer_type,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(LayerWithAttention, self).__init__()
        self.conv = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.gate = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.attention_func = attention_func

    def forward(self, x):
        c = self.conv(x)
        c = sample_norm(c)
        g = self.gate(x)
        g = sample_norm(g)
        out = F.tanh(c) * self.attention_func(g)
        return out


class ConvAttentionLayer(LayerWithAttention):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(ConvAttentionLayer, self).__init__(
            nn.Conv1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func)


class ConvTransposeAttentionLayer(LayerWithAttention):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(ConvTransposeAttentionLayer, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func)


class GeneratorWithAttention(nn.Module):
    def __init__(self):
        super(GeneratorWithAttention, self).__init__()
        self.attn_func = F.sigmoid
        self.layers = [
            ConvTransposeAttentionLayer(
                latent_dim, 512, 4, 2, 0, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 256, 8, 4, 2, attention_func=self.attn_func),
        ]
        self.main = nn.Sequential(*self.layers)
        self.final = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)
        self.gate = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        x = self.main(x)
        c = self.final(x)
        g = self.gate(x)
        x = F.sigmoid(g) * c
        return x


class Critic(nn.Module):
    def __init__(self, input_channels=1):
        super(Critic, self).__init__()
        self.input_channels = input_channels
        self.last_dim = 512
        self.activation = F.elu
        self.layers = [
            nn.Conv1d(input_channels, 512, 16, 8, 4),
            nn.Conv1d(512, 512, 8, 4, 2),
            nn.Conv1d(512, 512, 8, 4, 2),
            nn.Conv1d(512, 512, 8, 4, 2),
            nn.Conv1d(512, 512, 8, 4, 2),
            nn.Conv1d(512, 512, 4, 2, 0),
        ]
        self.main = nn.Sequential(*self.layers)
        self.linear = nn.Linear(self.last_dim, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, self.input_channels, sample_size)
        for m in self.main:
            x = m(x)
            x = self.activation(x)
        x = x.view(-1, self.last_dim)
        x = self.linear(x)
        return x


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = GeneratorWithAttention()
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
        needs=(scaled, wgan),
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
    from zounds.spectral import stft, rainbowgram
    samples = fake_audio()
    wscheme = zounds.SampleRate(
        frequency=samples.samplerate.frequency * 128,
        duration=samples.samplerate.frequency * 256)
    coeffs = stft(samples, wscheme, zounds.HanningWindowingFunc())
    return rainbowgram(coeffs)


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

    with app.start_in_thread(8888):
        if not Gan.exists():
            network = GanPair()

            for p in network.parameters():
                p.data.normal_(0, 0.02)

            trainer = zounds.WassersteinGanTrainer(
                network,
                latent_dimension=(latent_dim,),
                n_critic_iterations=10,
                epochs=500,
                batch_size=32,
                on_batch_complete=batch_complete)

            Gan.process(
                samples=(snd.windowed for snd in Sound),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float32)

        network = Gan().pipeline[-1].network.generator


        def sample_from_network():
            z = np.random \
                .normal(0, 1, (1, latent_dim, 1)).astype(np.float32)
            x = apply_network(network, z).reshape((1, sample_size))
            fake_samples[0] = x
            return fake_audio()

    app.start(8888)

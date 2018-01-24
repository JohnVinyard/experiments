from __future__ import division
import featureflow as ff
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import zounds
from zounds.learn import from_var
from random import choice

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

# window size in raw audio samples
sample_size = 8192

# size of the latent noise vector
latent_dim = 100

# the number of channels used for dilated convolutions
block_channels = 6

# the number of dilated layers
n_layers = 4

# the growth rate of channels in the "expansion" block of the
# critic/discriminator
growth_rate = 2

# the size of kernels in the "expansion" and "contraction" blocks of the critic
# discriminator, respectively
expand_contract_stride = 16

window_sample_rate = zounds.SampleRate(
    frequency=samplerate.frequency * sample_size,
    duration=samplerate.frequency * sample_size)


@zounds.simple_lmdb_settings(
    'wavenet_gan', map_size=1e11, user_supplied_id=True)
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
    if x.shape[1] == 1:
        return x

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


class DilatedBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            padding=0,
            do_sample_norm=False):
        super(DilatedBlock, self).__init__()
        self.do_sample_norm = do_sample_norm
        self.in_channels = in_channels
        self.dilation = dilation
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = self._conv(self.in_channels, kernel_size=2)

    def _conv(self, in_channels, kernel_size):
        return nn.Conv1d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = sample_norm(y)
        y = F.leaky_relu(y, 0.2)
        return y[..., :x.shape[-1]].contiguous()


class Stack(nn.Module):
    def __init__(
            self,
            in_channels,
            block_channels,
            n_layers,
            input_size,
            do_sample_norm=False):

        super(Stack, self).__init__()
        self.do_sample_norm = do_sample_norm
        self.input_size = input_size
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.n_layers = n_layers
        self.main = nn.Sequential(*self._build_dilated_blocks())

    @property
    def out_channels(self):
        if self.n_layers == 0:
            return self.in_channels
        else:
            return self.block_channels

    def _build_dilated_blocks(self):
        for i in xrange(self.n_layers):
            dilation = 2 ** i
            if dilation >= self.input_size:
                raise ValueError('dilation may not exceed total input size')
            in_channels = self.in_channels if i == 0 else self.block_channels
            yield DilatedBlock(
                in_channels=in_channels,
                out_channels=self.block_channels,
                dilation=dilation,
                padding=(dilation // 2) + 1,
                do_sample_norm=self.do_sample_norm)

    def forward(self, x):
        if self.n_layers == 0:
            return x
        x = x.view(-1, self.in_channels, self.input_size)
        for m in self.main:
            x = m(x) + x
        y = x
        return y


class Critic(nn.Module):
    def __init__(
            self,
            block_channels=block_channels,
            n_layers=n_layers,
            input_size=sample_size,
            growth_rate=growth_rate):
        super(Critic, self).__init__()
        self.stride = expand_contract_stride
        self.growth_rate = growth_rate
        self.input_size = input_size
        self.block_channels = block_channels
        if n_layers:
            self.stack = Stack(1, block_channels, n_layers, input_size)
        else:
            self.stack = None
        self.reduction = nn.Sequential(*self._reduce())
        self.linear = nn.Linear(
            list(self.reduction)[-1].out_channels, 1, bias=False)

    def _reduce(self):
        layers = 5
        channels = self.stack.out_channels
        new_channels = int(channels * self.growth_rate)

        yield nn.Conv1d(
            in_channels=channels,
            out_channels=new_channels,
            kernel_size=self.stride // 4,
            stride=self.stride // 8,
            padding=1,
            bias=False)
        channels = new_channels

        for i in xrange(layers - 1):
            new_channels = int(channels * self.growth_rate)
            yield nn.Conv1d(
                in_channels=channels,
                out_channels=new_channels,
                kernel_size=self.stride,
                stride=self.stride // 2,
                padding=self.stride // 4,
                bias=False)
            channels = new_channels

    def forward(self, x):
        if self.stack is not None:
            x = self.stack(x)
            channels = self.stack.out_channels
        else:
            channels = 1

        x = x.view(-1, channels, sample_size)
        for r in self.reduction:
            x = r(x)
            x = F.leaky_relu(x, 0.2)

        x = x.view(-1, self.linear.in_features)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim=latent_dim,
            block_channels=block_channels,
            n_layers=n_layers,
            input_size=sample_size):

        super(Generator, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.stride = expand_contract_stride
        self.expansion = nn.Sequential(*self._grow())
        if n_layers:
            self.stack = Stack(
                block_channels,
                block_channels,
                n_layers,
                input_size,
                do_sample_norm=True)
        else:
            self.stack = None
        self.conv = nn.Conv1d(
            in_channels=block_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)

    def _grow(self):
        layers = 5
        channels = self.latent_dim
        channel_step = (self.latent_dim - block_channels) // layers
        for i in xrange(layers - 1):
            new_channels = channels - channel_step
            yield nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=new_channels,
                kernel_size=self.stride,
                stride=self.stride // 2,
                padding=self.stride // 4,
                bias=False)
            channels = new_channels

        yield nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=block_channels,
            kernel_size=self.stride // 4,
            stride=self.stride // 8,
            padding=1,
            bias=False)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        for e in self.expansion:
            x = e(x)
            x = sample_norm(x)
            x = F.leaky_relu(x, 0.2)

        if self.stack is not None:
            x = self.stack(x)

        x = self.conv(x)
        return x


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = Generator()
        self.discriminator = Critic()


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
    fake_samples[0] = from_var(samples).squeeze()


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
                batch_size=12,
                on_batch_complete=batch_complete)

            Gan.process(
                samples=gen(),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float32)

    app.start(8888)

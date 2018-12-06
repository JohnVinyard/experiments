from __future__ import division
import zounds
import featureflow as ff
from zounds.learn import Conv1d, ConvTranspose1d, Conv2d, to_var, from_var
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
from random import choice
from scipy.fftpack import dct
import torch
from torch.autograd import Variable

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


def dct_basis(size):
    r = np.arange(size)
    basis = np.outer(r, r + 0.5)
    basis = np.cos((np.pi / size) * basis)
    return basis


basis = dct_basis(512)
basis *= zounds.HanningWindowingFunc()
basis = torch.from_numpy(basis.astype(np.float32)).unsqueeze(1)
basis = Variable(basis).cuda()

scale = zounds.GeometricScale(
    start_center_hz=50,
    stop_center_hz=samplerate.nyquist,
    bandwidth_ratio=0.035,
    n_bands=300)
scale.ensure_overlap_ratio(0.5)


def spiky(n):
    if n == 1:
        return np.ones(n)

    output = np.zeros(n)
    if n % 2 == 1:
        output[(n // 2)] = 1
    else:
        output[(n // 2) - 1: (n // 2) + 1] = 0.5
    return output


linear_scale = zounds.LinearScale.from_sample_rate(samplerate, 512)
geometric_basis = scale._basis(
    linear_scale, zounds.WindowingFunc(spiky))
geometric_basis *= zounds.HanningWindowingFunc()
geometric_basis = torch.from_numpy(
    geometric_basis.astype(np.float32)).unsqueeze(2)
geometric_basis = Variable(geometric_basis).cuda()


def dct_transform(samples):
    pt_linear_coeffs = F.conv1d(
        samples, basis, stride=basis.shape[0] // 2)

    # norms = torch.norm(pt_linear_coeffs, dim=1)
    # pt_linear_coeffs = pt_linear_coeffs / (norms.unsqueeze(1) + 1e12)

    # scaling = Variable(torch.FloatTensor(1)).cuda()
    # scaling[:] = 2. / 512
    # pt_linear_coeffs *= torch.sqrt(scaling)




    # pt_geometric_coeffs = F.conv1d(pt_linear_coeffs, geometric_basis)
    # pt_final = torch.log(torch.abs(pt_geometric_coeffs * 10) + 1)
    # return pt_final
    return pt_linear_coeffs


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ConvTranspose1d(
                latent_dim, 512, 4, 1, 0,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 512, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 512, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 512, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 512, 8, 4, 2,
                sample_norm=True, dropout=False, activation=activation),
            ConvTranspose1d(
                512, 1, 16, 8, 4,
                sample_norm=False,
                batch_norm=False,
                dropout=False,
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
    def __init__(self):
        super(Critic, self).__init__()
        self.last_dim = 512
        self.activation = F.elu
        # self.raw = Conv1d(
        #     1, 512, 512, 256, 0,
        #     batch_norm=False, dropout=False, activation=self.activation)

        self.main = nn.Sequential(
            Conv1d(
                512, 512, 1, 1, 1,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                512, 512, 4, 2, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                512, 512, 4, 2, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                512, 512, 4, 2, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                512, 512, 4, 2, 2,
                batch_norm=False, dropout=False, activation=self.activation),
            Conv1d(
                512, self.last_dim, 3, 1, 0,
                batch_norm=False, dropout=False, activation=self.activation)
        )
        self.linear = nn.Linear(self.last_dim, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, sample_size)

        # raw = self.raw(x)
        x = dct_transform(x)

        # x = torch.cat([raw, x], dim=1)

        for m in self.main:
            x = m(x)

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
        zounds.PhatDrumLoops(),
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
            yield sound.windowed
            # start = zounds.Seconds(10)
            # end = sound.windowed.dimensions[0].end - zounds.Seconds(10)
            # duration = end - start
            # ts = zounds.TimeSlice(start=start, duration=duration)
            # yield sound.windowed[ts]


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
                samples=gen(),
                trainer=trainer,
                nsamples=int(5e5),
                dtype=np.float32)

    app.start(8888)

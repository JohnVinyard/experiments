import zounds
from torch.optim import Adam
from encoder import Encoder
from generator import *
from upsample import \
    NearestNeighborUpsamplingBlock, LinearUpSamplingBlock, DctUpSamplingBlock
from itertools import chain
from zounds.learn import PerceptualLoss
import numpy as np
from torch.autograd import Variable
from scipy.signal import gaussian
from loss import BandLoss2
from torch.nn.init import xavier_normal, orthogonal, calculate_gain

samplerate = zounds.SR11025()
window_size = 8192
hop_size = 128
latent_dim = 128
n_filters = 32
exp = None

wscheme = zounds.SampleRate(
    frequency=samplerate.frequency * hop_size,
    duration=samplerate.frequency * window_size)

BaseModel = zounds.windowed(
    wscheme=wscheme, resample_to=samplerate, store_resampled=True)


@zounds.simple_lmdb_settings('generator', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    scaled = zounds.ArrayWithUnitsFeature(
        lambda x: zounds.instance_scale(x, axis=0),
        needs=BaseModel.windowed)


class Experiment(object):
    def __init__(self, samples, encoder, generator, loss, iterations):
        super(Experiment, self).__init__()
        self.iterations = iterations
        self.samples = samples
        self.encoder = encoder
        self.generator = generator
        self.loss = loss
        self.optimizer = Adam(
            chain(encoder.parameters(), generator.parameters()),
            lr=0.00001,
            betas=(0, 0.9))
        self.decoded = None

    def _init_network(self, n):
        for p in n.parameters():
            if p.dim() >= 2:
                p.data = xavier_normal(
                    p.data, gain=calculate_gain('leaky_relu', param=0.2))

    @property
    def real_samples(self):
        return zounds.instance_scale(self.samples).pad_with_silence()

    @property
    def fake_samples(self):
        return zounds.instance_scale(self.decoded).pad_with_silence()

    @property
    def parameter_count(self):
        return self.generator.parameter_count

    def run(self):
        self.encoder = self.encoder.cuda()
        self.generator = self.generator.cuda()
        self._init_network(self.encoder)
        self._init_network(self.generator)

        for i in xrange(self.iterations):
            self.encoder.zero_grad()
            self.generator.zero_grad()
            batch = Variable(torch.from_numpy(self.samples)).cuda()
            encoded = self.encoder(batch)
            decoded = self.generator(encoded)
            self.decoded = zounds.AudioSamples(
                decoded.data.cpu().numpy().squeeze(),
                self.samples.samplerate)
            error = self.loss(decoded, batch)
            error.backward()
            print i, self.generator, error.data[0]
            self.optimizer.step()

        return str(self.generator), error.data[0], self.parameter_count


def networks(latent_dim, window_size, n_filters):
    ks = [3, 7, 15, 31]
    dilation_sizes = [1, 2, 4, 8, 16]
    scale = zounds.MelScale(
        zounds.FrequencyBand(20, samplerate.nyquist - 300), 500)

    filters = zounds.fir_filter_bank(scale, 512, samplerate, gaussian(100, 3))
    filters = Variable(torch.from_numpy(filters).float()).cuda()
    filters = filters.view(len(scale), 1, 512).contiguous()

    yield AllMultiResGenerator(
        latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock,
        [7, 15, 31, 63])

    yield AllMultiResGenerator(
        latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock,
        [31, 63, 127])

    yield AllMultiResGenerator(
        latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks)

    yield NoiseTransformer(
        latent_dim, window_size, n_filters)



    # yield DeepUpSamplingGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock)
    #
    # yield MultiBranch(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks,
    #     final_activation=torch.cos, inner_activation=torch.cos)
    #
    # yield AllMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks,
    #     final_activation=torch.cos)
    #
    # yield AllMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks,
    #     final_activation=torch.cos, inner_activation=torch.cos)
    #
    # yield AllMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks,
    #     final_activation=lambda x: x, inner_activation=torch.cos)

    # yield LearnedFirstLayerMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks)
    #
    # yield AllMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks)
    #
    # yield UpSamplingMultiResGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks)
    #
    # yield UpSamplingMultiResGeneratorFinalFrozenLayer(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks,
    #     filters)
    #
    # yield FrozenFinalLayer(latent_dim, window_size, n_filters, filters)
    #
    # yield PsychoAcousticMultiDilationUpSamplingGenerator(
    #     latent_dim,
    #     window_size,
    #     n_filters,
    #     NearestNeighborUpsamplingBlock,
    #     dilation_sizes)
    #
    # yield PsychoAcousticMultiResUpSamplingGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock, ks)
    #

    #
    yield MultiScaleUpSamplingGenerator(
        latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock)
    #
    # yield ConvGenerator(latent_dim, window_size, n_filters)
    #
    # yield PschyoAcousticUpsamplingGenerator(
    #     latent_dim, window_size, n_filters, NearestNeighborUpsamplingBlock)
    #
    # yield MultiScaleGenerator(latent_dim, window_size, n_filters)


if __name__ == '__main__':
    zounds.ingest(
        zounds.InternetArchive('LucaBrasi2'),
        Sound,
        multi_threaded=True)

    scale = zounds.MelScale(
        frequency_band=zounds.FrequencyBand(20, samplerate.nyquist - 300),
        n_bands=500)

    loss = PerceptualLoss(
        scale,
        samplerate,
        lap=1,
        log_factor=10,
        frequency_weighting=zounds.AWeighting(),
        phase_locking_cutoff_hz=1200).cuda()

    encoder = Encoder(latent_dim, n_filters)

    iterations = 2000

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.scaled,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())

    with app.start_in_thread(8888):
        while True:

            snd = Sound.random()
            index = np.random.randint(0, len(snd.scaled))
            samples = zounds.AudioSamples(snd.scaled[index], samplerate)

            results = []

            for generator in networks(latent_dim, window_size, n_filters):
                exp = Experiment(samples, encoder, generator, loss, iterations)
                name, error, params = exp.run()
                results.append((name, error, params))

            descending_param_count = sorted(
                results, key=lambda x: x[2], reverse=True)

            for result in descending_param_count:
                print result

            raw_input('Start another round?')

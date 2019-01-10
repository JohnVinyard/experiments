from __future__ import print_function, division
import zounds
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_, calculate_gain
from torch.nn import functional as F
import torch
import argparse
from torch.optim import Adam
from scipy.signal import morlet
from zounds.learn import gradients


samplerate = zounds.SR11025()

BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)
window_size_samples = 8192

channels = 128
band = zounds.FrequencyBand(1, samplerate.nyquist)
scale = zounds.MelScale(band, channels)


@zounds.simple_lmdb_settings('sounds', 1e11, user_supplied_id=True)
class Sound(BaseModel):
    pass


def apply_dilated_stack(x, main, gate, debug=False):
    accumulator = None
    for m, g in zip(main, gate):
        y = F.tanh(m(x)) * F.sigmoid(g(x))
        if debug:
            print(x.shape, y.shape)
        if x.shape[1:] == y.shape[1:]:
            x = x + y
        else:
            x = y

        if accumulator is None:
            accumulator = x
        else:
            accumulator = accumulator + x

    return accumulator


def dilated_stack(in_channels, channels, max_dilation, twod=False):
    dilation = 1
    layers = []
    while dilation <= max_dilation:
        cls = nn.Conv2d if twod else nn.Conv1d
        layer = cls(
            in_channels if dilation == 1 else channels,
            channels,
            kernel_size=(2, 1) if twod else 2,
            stride=(1, 1) if twod else 1,
            dilation=(dilation, 1) if twod else dilation,
            padding=(dilation // 2, 0) if twod else dilation // 2 or 1,
            bias=False)
        layers.append(layer)
        dilation *= 2
    return nn.Sequential(*layers)


def make_filter_bank(samplerate, kernel_size, scale):
    """
    Create a bank of finite impulse response filters, with
    frequencies centered on the sub-bands of scale
    """
    basis = np.zeros((len(scale), kernel_size), dtype=np.complex128)

    for i, band in enumerate(scale):
        # cycles determines the tradeoff between time and frequency
        # resolution.  We'd like good frequency resolution for lower
        # frequencies, and good time resolution for higher ones
        cycles = max(64, int(samplerate) / band.center_frequency)
        basis[i] = morlet(
            kernel_size,  # wavelet size
            cycles,  # time-frequency resolution tradeoff
            (band.center_frequency / samplerate.nyquist))  # frequency
    return basis.real


def make_filter_bank2(kernel_size, n_bands):
    basis = np.zeros((n_bands, kernel_size), dtype=np.complex128)
    scale = np.geomspace(0.5, 0.9, n_bands)
    for i, center_frequency in enumerate(scale):
        basis[i] = morlet(
            kernel_size, 32, center_frequency)
    return basis.real


class FilterBank(nn.Module):
    def __init__(self, samplerate, kernel_size, scale):
        super(FilterBank, self).__init__()

        # filter_bank = make_filter_bank(samplerate, kernel_size, scale)
        filter_bank = make_filter_bank2(kernel_size, len(scale))

        self.scale = scale
        self.filter_bank = torch.from_numpy(filter_bank).float() \
            .view(len(scale), 1, kernel_size)
        self.filter_bank.requires_grad = False

    def to(self, *args, **kwargs):
        self.filter_bank = self.filter_bank.to(*args, **kwargs)
        return super(FilterBank, self).to(*args, **kwargs)

    def convolve(self, x):
        x = x.view(-1, 1, x.shape[-1])
        x = F.conv1d(
            x, self.filter_bank, padding=self.filter_bank.shape[-1] // 2)
        return x

    def transposed_convolve(self, x):
        x = F.conv_transpose1d(
            x, self.filter_bank, padding=self.filter_bank.shape[-1] // 2)
        return x

    def log_magnitude(self, x):
        x = F.relu(x)
        x = 20 * torch.log10(1 + x)
        return x

    def temporal_pooling(self, x, kernel_size, stride):
        x = F.avg_pool1d(x, kernel_size, stride, padding=kernel_size // 2)
        return x

    def normalize(self, x):
        """
        give each instance unit norm
        """
        orig_shape = x.shape
        x = x.view(x.shape[0], -1)

        x = x / torch.norm(x, dim=-1, keepdim=True) + 1e-8

        x = x.view(orig_shape)
        return x

    def transform(self, samples, pooling_kernel_size, pooling_stride):
        # convert the raw audio samples to a PyTorch tensor
        tensor_samples = torch.from_numpy(samples).float() \
            .to(self.filter_bank.device)

        # compute the transform
        spectral = self.convolve(tensor_samples)
        log_magnitude = self.log_magnitude(spectral)
        pooled = self.temporal_pooling(
            log_magnitude, pooling_kernel_size, pooling_stride)

        # convert back to an ArrayWithUnits instance
        samplerate = samples.samplerate
        time_frequency = pooled.data.cpu().numpy().squeeze().T
        time_frequency = zounds.ArrayWithUnits(time_frequency, [
            zounds.TimeDimension(
                frequency=samplerate.frequency * pooling_stride,
                duration=samplerate.frequency * pooling_kernel_size),
            zounds.FrequencyDimension(self.scale)
        ])
        return time_frequency

    def forward(self, x, log_magnitude=True, normalize=True):
        nsamples = x.shape[-1]
        x = self.convolve(x)

        if log_magnitude:
            x = self.log_magnitude(x)

        if normalize:
            x = self.normalize(x)

        return x[..., :nsamples].contiguous()


class AutoEncoder(nn.Module):
    def __init__(
            self,
            size,
            channels,
            latent_dim,
            activation=F.elu,
            filter_bank=None):

        super(AutoEncoder, self).__init__()
        self.filter_bank = filter_bank
        self.latent_dim = latent_dim
        self.activation = activation
        self.size = size
        self.channels = channels


        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (1, 3), (1, 2), padding=(0, 1), bias=False),

            # nn.Conv2d(8, 8, (1, 3), (1, 2), padding=(0, 1), bias=False),
            # nn.Conv2d(8, 8, (3, 1), (2, 1), padding=(1, 0), bias=False),

            # This is where things go bad!
            # nn.Conv2d(8, 512, (1, 3), (1, 2), padding=(0, 1), bias=False),

        )

        # self.linear = nn.Sequential(
        #     nn.Linear(1024, 512, bias=False),
        #     nn.Linear(512, 256, bias=False),
        #     nn.Linear(256, 128, bias=False),
        # )
        #
        # self.final = nn.Linear(128, self.latent_dim, bias=False)
        #
        # self.expand = nn.Sequential(
        #     nn.Linear(self.latent_dim, 128, bias=False),
        #     nn.Linear(128, 256, bias=False),
        #     nn.Linear(256, 512, bias=False),
        #     nn.Linear(512, 1024, bias=False),
        # )

        self.decoder = nn.Sequential(


            # This is where things go bad
            # nn.ConvTranspose2d(512, 8, (1, 3), (1, 2), padding=(0, 0), bias=False),

            # nn.ConvTranspose2d(8, 8, (3, 1), (2, 1), padding=(0, 0), bias=False),
            # nn.ConvTranspose2d(8, 8, (1, 3), (1, 2), padding=(0, 0), bias=False),

            nn.ConvTranspose2d(8, 1, (1, 3), (1, 2), padding=(0, 0), bias=False)
        )

        # self.synth_params = nn.Conv1d(channels, channels, 1, 1, padding=0, bias=False)
        # self.synth_params_gate = nn.Conv1d(channels, channels, 1, 1, padding=0, bias=False)

    def initialize_weights(self):
        for p in self.encoder.parameters():
            if p.data.dim() > 2:
                xavier_normal_(p.data, calculate_gain('leaky_relu', 0.2))

        for p in self.decoder.parameters():
            if p.data.dim() > 2:
                xavier_normal_(p.data, calculate_gain('leaky_relu', 0.2))

        # for p in self.linear.parameters():
        #     if p.data.dim() > 2:
        #         xavier_normal_(p.data, calculate_gain('leaky_relu', 0.2))
        #
        # for p in self.expand.parameters():
        #     if p.data.dim() > 2:
        #         xavier_normal_(p.data, calculate_gain('leaky_relu', 0.2))
        #
        # for p in self.final.parameters():
        #     if p.data.dim() > 2:
        #         xavier_normal_(p.data, 1)
        #
        # for p in self.synth_params.parameters():
        #     if p.data.dim() > 2:
        #         xavier_normal_(p.data, calculate_gain('tanh'))
        #
        # for p in self.synth_params_gate.parameters():
        #     if p.data.dim() > 2:
        #         xavier_normal_(p.data, calculate_gain('sigmoid'))

    def encode(self, x):
        x = x.view(-1, 1, self.size)
        x = self.filter_bank(x)
        x = x[:, None, ...]

        for e in self.encoder:
            x = e(x)
            x = self.activation(x)
            x = x.squeeze(dim=-2)
            # print('encoded', x.shape)

        # x = x.squeeze(dim=-1)
        # for l in self.linear:
        #     x = l(x)
        #     x = self.activation(x)
        #
        # x = self.final(x)
        return x

    def decode(self, x):
        # for e in self.expand:
        #     x = e(x)
        #     x = self.activation(x)

        # x = x[..., None]

        for d in self.decoder:
            try:
                x = d(x)
            except RuntimeError:
                # we should now be doing 2d convolution
                x = x[:, :, None, :]
                x = d(x)
            x = self.activation(x)
            # print('decoded', x.shape)

        x = x.squeeze(dim=1)
        x = x[:, :self.channels, :self.size]

        # x = F.tanh(self.synth_params(x)) * F.sigmoid(self.synth_params_gate(x))

        x = self.filter_bank.transposed_convolve(x)
        x = F.pad(x, (0, 1), mode='constant', value=0)

        x = x[..., :self.size]
        x = x / torch.norm(x, dim=-1, keepdim=True) + 1e-8

        # print('final decoded', x.shape)
        return x

    def forward(self, x):
        encoded = self.encode(x)
        # print('final encoded', encoded.shape)
        decoded = self.decode(encoded)
        return decoded


class Sampler(object):
    def __init__(self, sound_cls, slice_duration):
        super(Sampler, self).__init__()
        self.slice_duration = slice_duration
        self.sound_cls = sound_cls
        items = [(snd._id, snd.resampled.end) for snd in self.sound_cls]
        durations = np.array([item[1] / zounds.Seconds(1) for item in items])
        probabilities = durations / durations.sum()
        self.probabilities = probabilities
        self.items = [item[0] for item in items]
        self.durations = dict(items)

    def _get_samples(self, sound_id, start_ps):
        snd = self.sound_cls(_id=sound_id)
        start = zounds.Picoseconds(int(start_ps))
        time_slice = zounds.TimeSlice(start=start, duration=self.slice_duration)
        max_samples = \
            int(self.slice_duration / snd.resampled.samplerate.frequency)
        return time_slice, snd.resampled[time_slice][:max_samples]

    def _sample_slice(self):
        _id = np.random.choice(self.items, p=self.probabilities)
        duration = self.durations[_id]
        duration_ps = duration / zounds.Picoseconds(1)
        slice_duration_ps = self.slice_duration / zounds.Picoseconds(1)
        start_ps = np.random.uniform(0, duration_ps - slice_duration_ps)
        time_slice, samples = self._get_samples(_id, start_ps)
        return time_slice, samples

    def sample(self, batch_size):
        samples = [self._sample_slice()[1] for _ in xrange(batch_size)]
        time_dimension = samples[0].dimensions[-1]
        samples = np.stack(samples, axis=0)
        samples = zounds.ArrayWithUnits(
            samples, [zounds.IdentityDimension(), time_dimension])
        return torch.from_numpy(samples).float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.ui.AppSettings()
    ])
    parser.add_argument(
        '--ingest',
        help='Should audio be ingested',
        action='store_true')
    parser.add_argument(
        '--batch-size',
        help='batch size',
        type=int)
    parser.add_argument(
        '--test',
        help='test the network',
        action='store_true')

    args = parser.parse_args()

    if args.ingest:
        dataset = zounds.InternetArchive('LucaBrasi2')
        zounds.ingest(dataset, Sound, multi_threaded=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filter_bank = FilterBank(samplerate, 64, scale).to(device)

    network = AutoEncoder(
        size=512,
        channels=128,
        latent_dim=64,
        filter_bank=filter_bank,
        activation=F.elu).to(device)

    try:
        state_dict = torch.load('weights.dat')
        network.load_state_dict(state_dict)
        print('loaded weights')
    except IOError:
        network.initialize_weights()
        print('initialized weights')

    sampler = Sampler(Sound, window_size_samples * samplerate.frequency)

    transform = zounds.learn.DctTransform(use_cuda=torch.cuda.is_available())

    if args.test:
        test_input = sampler.sample(args.batch_size).to(device)

        bands = transform.frequency_decomposition(
            test_input, [1, 0.5, 0.25, 0.125, 0.0625])
        segments = torch.cat(
            [band.view(-1, 512) for band in bands])

        output = network(segments)
        print('OUTPUT', output.shape)
        exit()

    optimizer = Adam(
        [p for p in network.parameters() if p.requires_grad],
        lr=1e-6)

    loss = nn.MSELoss()

    batch = None
    decoded = None

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.resampled,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())

    app.start_in_thread(args.port)


    def synthesize():
        with torch.no_grad():
            size = 512

            orig = sampler.sample(1)
            batch = orig.to(device)

            scales = [1, 0.5, 0.25, 0.125, 0.0625][::-1]
            bands = transform.frequency_decomposition(batch, scales)

            segments = torch.cat([band.view(-1, size) for band in bands])
            norms = torch.norm(segments, dim=-1, keepdim=True) + 1e-8
            segments /= norms

            encoded = network.encode(segments)
            decoded = network.decode(encoded).squeeze()
            decoded_normed = \
                decoded / torch.norm(decoded, dim=-1, keepdim=True) + 1e-8
            normed = decoded_normed * norms

            final = zounds.AudioSamples.silence(
                samplerate, window_size_samples * samplerate.frequency)

            current_pos = 0
            for scale in scales:
                total_size = int(scale * window_size_samples)
                current_size = total_size // size
                band = normed[current_pos:current_pos + current_size]
                band = band.view(1, total_size)
                upsampled = transform.dct_resample(
                    band, window_size_samples / band.shape[-1])
                current_pos += current_size
                final += upsampled.squeeze()

            # check = torch.from_numpy(final).to(device).view(1, 1, len(final))
            # check_bands = transform.frequency_decomposition(check, scales)
            # segments = torch.cat([band.view(-1, size) for band in check_bands])
            # check_norms = torch.norm(segments, dim=-1, keepdim=True) + 1e-8

            return \
                zounds.AudioSamples(orig.data.cpu().numpy().squeeze(), samplerate).pad_with_silence(), \
                final.pad_with_silence()

    fb = filter_bank.filter_bank.data.cpu().numpy().squeeze()

    while True:
        batch = sampler.sample(args.batch_size).to(device)
        bands = transform.frequency_decomposition(
            batch, [1, 0.5, 0.25, 0.125, 0.0625])

        segments = torch.cat(
            [band.view(-1, 512) for band in bands])

        # shuffle the batch
        segments = segments[torch.randperm(segments.shape[0])]

        # give each segment unit norm
        norms = torch.norm(segments, dim=-1, keepdim=True) + 1e-8
        segments = segments / norms

        with torch.no_grad():
            f = filter_bank(segments).data.cpu().numpy().squeeze()

        # encode/decode round trip
        encoded = network.encode(segments)
        e = encoded.data.cpu().numpy().squeeze()
        decoded = network.decode(encoded)

        with torch.no_grad():
            f2 = filter_bank(decoded).data.cpu().numpy().squeeze()

        # perceptual features
        real = filter_bank(segments)
        fake = filter_bank(decoded)

        # compute loss
        error = loss(fake, real)

        # for g in gradients(network):
        #     print(g)

        try:
            error.backward()
            optimizer.step()
        except RuntimeError:
            pass

        print(error.item())

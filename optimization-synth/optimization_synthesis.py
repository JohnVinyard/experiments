from __future__ import print_function, division
import zounds
from torch.optim import Adam
from torch import nn
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from torch.nn.init import calculate_gain, xavier_normal_, orthogonal_
import math
from zounds.learn.util import batchwise_unit_norm


samplerate = zounds.SR11025()
BaseModel = zounds.stft(
    resample_to=samplerate,
    store_resampled=True)


@zounds.simple_lmdb_settings('optsynth', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    pass


dataset = zounds.CompositeDataset(
    # Classical
    zounds.InternetArchive('AOC11B'),
    zounds.InternetArchive('beethoven_ingigong_850'),
    zounds.InternetArchive('The_Four_Seasons_Vivaldi-10361'),

    # Pop
    zounds.InternetArchive('02.LostInTheShadowsLouGramm'),
    zounds.InternetArchive('08Scandalous'),

    # Jazz
    zounds.InternetArchive('Free_20s_Jazz_Collection'),
    zounds.InternetArchive('JohnColtrane-GiantSteps'),

    # Hip Hop
    zounds.InternetArchive('LucaBrasi2'),
    zounds.InternetArchive('Chance_The_Rapper_-_Coloring_Book'),

    # Speech
    zounds.InternetArchive('Greatest_Speeches_of_the_20th_Century'),

    # Electronic
    zounds.InternetArchive('rome_sample_pack'),
    zounds.InternetArchive('CityGenetic'),
    zounds.InternetArchive('HalloweenStickSynthRaver'),

    # Nintendo
    zounds.InternetArchive('CastlevaniaNESMusicStage10WalkingOnTheEdge'),
    zounds.InternetArchive('JourneyToSiliusNESMusicStageTheme01')
)


class OptiSynth(object):
    def __init__(
            self,
            analyzer,
            synthesizer,
            synth_params,
            loss,
            n_iterations=10000):
        super(OptiSynth, self).__init__()
        self.loss = loss
        self.n_iterations = n_iterations
        self.synth_params = synth_params
        self.synthesizer = synthesizer
        self.analyzer = analyzer
        self.current_synth_samples = None
        self.features = None

    @property
    def best(self):
        return self.current_synth_samples

    @property
    def fft(self):
        return np.abs(zounds.spectral.stft(self.best))

    @property
    def rainbow(self):
        fft = zounds.spectral.stft(self.best)
        return zounds.spectral.rainbowgram(fft)

    @property
    def fine(self):
        # KLUDGE: This assumes the FIR analyzer
        return zounds.log_modulus(
            np.abs(np.fft.rfft(self.features[:, 1024:1024 + 256])))

    def __call__(self, samples):
        optim = Adam([self.synth_params], lr=.0001)
        real_features = self.analyzer(samples)
        real_features = real_features.detach()
        self.features = real_features.data.cpu().numpy().squeeze()

        for iteration in xrange(self.n_iterations):
            synth_samples = self.synthesizer(self.synth_params)
            self.current_synth_samples = zounds.AudioSamples(
                synth_samples.data.cpu().numpy().squeeze(),
                samplerate).pad_with_silence()
            synth_features = self.analyzer(synth_samples)
            print(real_features.shape, synth_features.shape)
            error = self.loss(synth_features, real_features)
            error.backward()
            print(iteration, error.item())
            optim.step()


class RawSampleSynth(nn.Module):
    def __init__(self):
        super(RawSampleSynth, self).__init__()

    def forward(self, x):
        return x


class STFTAnalyzer(nn.Module):
    def __init__(self):
        super(STFTAnalyzer, self).__init__()

    def forward(self, x):
        x = x.view(-1, x.shape[-1])
        window = torch.hann_window(256).to(x.device)
        return torch.stft(
            x,
            frame_length=256,
            hop=256,
            fft_size=512,
            normalized=True,
            window=window)[..., 0]


class SelfSimilarityAnalyzer(nn.Module):
    def __init__(self):
        super(SelfSimilarityAnalyzer, self).__init__()

    def forward(self, x):
        x = x.squeeze()
        x = torch.ger(x, x)
        return x


def apply_dilated_stack(x, main, gate):
    accumulator = None
    for m, g in zip(main, gate):
        y = F.tanh(m(x)) * F.sigmoid(g(x))
        if x.shape[1:] == y.shape[1:]:
            x = x + y
        else:
            x = y

        if accumulator is None:
            accumulator = x
        else:
            accumulator = accumulator + x

    return accumulator


class EmbeddingNetwork7(nn.Module):
    """
    BEST YET

    Simple, Wavenet-style architecture with global max pooling at the end
    """
    def __init__(self):
        super(EmbeddingNetwork7, self).__init__()

        channels = 128
        in_channels = 1

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, channels, 2, 1, dilation=1, padding=0, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=2, padding=1, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=4, padding=2, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=8, padding=4, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=16, padding=8, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=32, padding=16, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=64, padding=32, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=128, padding=64, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=256, padding=128, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=512, padding=256, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=1024, padding=512, bias=False),
        )

        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, channels, 2, 1, dilation=1, padding=0),
            nn.Conv1d(channels, channels, 2, 1, dilation=2, padding=1, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=4, padding=2, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=8, padding=4, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=16, padding=8, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=32, padding=16, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=64, padding=32, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=128, padding=64, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=256, padding=128, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=512, padding=256, bias=False),
            nn.Conv1d(channels, channels, 2, 1, dilation=1024, padding=512, bias=False),
        )

    def initialize_weights(self):
        for m in self.main.parameters():
            if m.data.dim() > 2:
                # xavier_normal_(m.data, calculate_gain('tanh'))
                orthogonal_(m.data, calculate_gain('tanh'))

        for m in self.gate.parameters():
            if m.data.dim() > 2:
                # xavier_normal_(m.data, calculate_gain('sigmoid'))
                orthogonal_(m.data, calculate_gain('sigmoid'))

    def forward(self, x):
        # normalize
        x = x.view(-1, 8192)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        x = x.view(-1, 1, 8192)
        x = apply_dilated_stack(x, self.main, self.gate)
        x = F.max_pool1d(x, x.shape[-1])
        x = x.squeeze(dim=-1)
        x = batchwise_unit_norm(x)
        return x

class FIRAnalyzer(nn.Module):
    def __init__(self, scale, fine_scale, taps, fine_taps):
        super(FIRAnalyzer, self).__init__()
        self.taps = taps
        self.scale = scale
        self.weights = self._filter_bank(scale, taps)
        self.fine_weights = F.relu(self._filter_bank(fine_scale, fine_taps))

    def _filter_bank(self, scale, taps):
        filter_bank = zounds.spectral.fir_filter_bank(
            scale, taps, samplerate, np.hanning(taps))
        return torch \
            .from_numpy(filter_bank) \
            .float() \
            .view(len(scale), 1, taps)

    def forward(self, x):
        x = x.view(-1, 1, window_size_samples)

        # first, compute a spectrogram by convolving the signal with a bank
        # of FIR filters
        x = F.conv1d(
            x, self.weights.to(x.device), stride=1, padding=self.taps // 2)
        x = F.relu(x)
        x = torch.log(1 + (x * 100))
        print(x.shape)

        x = F.conv1d(
            x,
            x.permute(1, 0, 2).contiguous(),
            padding=x.shape[-1] // 2,
            groups=x.shape[1])
        print(x.shape)

        # then compute within-band oscillations using another filter bank
        # x = (1, 512, 8192)
        # fine = (32, 1, 256)
        # x = F.conv2d(
        #     x[:, None, ...],
        #     self.fine_weights[:, None, ...].to(x.device))
        # print(x.shape)
        # x = F.max_pool2d(x, (1, 256), (1, 128))
        # print(x.shape)
        return x


class MultiScaleAnalyzer(nn.Module):
    def __init__(self):
        super(MultiScaleAnalyzer, self).__init__()
        self.dct = zounds.learn.DctTransform(use_cuda=True)
        self.scales = [0.0625, 0.125, 0.25, 0.5, 1]
        self.taps = [int(s * 512) for s in self.scales]

        filter_banks = []
        start_frequency = 0

        for scale, taps in zip(self.scales, self.taps):
            band = zounds.FrequencyBand(
                start_frequency, samplerate.nyquist * scale)
            mel_scale = zounds.MelScale(band, 32)
            bank = zounds.spectral.fir_filter_bank(
                mel_scale, taps, samplerate, np.hamming(100))
            bank = torch.from_numpy(bank).float().cuda()\
                .view(len(mel_scale), 1, taps)
            filter_banks.append(bank)

        self.filter_banks = filter_banks

        coarse_scale = zounds.MelScale(
            zounds.FrequencyBand(0, samplerate.nyquist), len(self.scales))
        weighting = zounds.AWeighting()
        self.frequency_weighting = weighting._wdata(coarse_scale)

    def forward(self, x):
        x = x.view(-1, 1, window_size_samples)
        bands = self.dct.frequency_decomposition(x, self.scales)
        features = []
        for band, bank, taps, weighting in zip(
                bands, self.filter_banks, self.taps, self.frequency_weighting):
            print(band.max().item())
            f = F.conv1d(band, bank, stride=1, padding=bank.shape[-1] // 2)
            f = F.relu(f)
            f = torch.log(1 + (10 * f))
            # f = f.unfold(-1, taps, taps // 2)
            # f = f * torch.hamming_window(f.shape[-1]).to(f.device)
            # f = torch.rfft(f, 1, normalized=True)[..., 0]
            features.append(f)

        x = torch.cat(features, dim=-1)
        return x


class MatchingPursuitSynth(nn.Module):
    def __init__(self, nsamples):
        super(MatchingPursuitSynth, self).__init__()
        self.nsamples = nsamples
        self.amps = None

    def forward(self, x):
        means = torch.abs(x[..., 0])
        stds = torch.abs(x[..., 1])
        freq = torch.abs(x[..., 2] * samplerate.nyquist)
        phase = torch.abs(x[..., 3] * (2 * math.pi))

        two_pi = 2 * math.pi

        amp_mod = torch.abs(x[..., 4] * samplerate.nyquist * 0.01)[..., None]
        amp_mod_phase = torch.abs(x[..., 5] * two_pi)[..., None]

        freq_mod = torch.abs(x[..., 6] * samplerate.nyquist * 0.01)[..., None]
        freq_mod_phase = torch.abs(x[..., 7] * two_pi)[..., None]

        base = torch.arange(0, 1.0, 1 / self.nsamples).to(x.device)
        two_pi = torch.FloatTensor(1).fill_(two_pi).to(x.device)
        u = (base[..., None] - means) / stds
        amp = \
            (1 / torch.sqrt(two_pi) * stds) * torch.exp(-u * (u / 2.))
        amp = amp.transpose(0, 1).contiguous()
        self.amps = amp.data.cpu().numpy().squeeze()

        sine_base = base * math.pi * 2
        # amp = amp * torch.sin(sine_base * amp_mod + amp_mod_phase)

        freq = freq[..., None]
        # freq = freq * torch.sin(sine_base * freq_mod + freq_mod_phase)

        phase = phase[..., None]
        elements = amp * torch.sin((sine_base * freq) + phase)
        final = torch.sum(elements, dim=1, keepdim=True)
        return final



# TODO: try pink noise
# TODO: try rainbowgram phase transformation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.AppSettings()
    ])
    parser.add_argument(
        '--ingest',
        help='flag indicating that audio should be ingested',
        action='store_true')
    args = parser.parse_args()

    if args.ingest:
        zounds.ingest(dataset, Sound, multi_threaded=True)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.fft,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    window_size_samples = 8192

    synth = RawSampleSynth()

    scale = zounds.MelScale(
        zounds.FrequencyBand(0, samplerate.nyquist), 128)
    fine_scale = zounds.MelScale(
        zounds.FrequencyBand(0, samplerate.nyquist), 16)

    # analyzer = FIRAnalyzer(
    #     scale, fine_scale, taps=256, fine_taps=256).to(device)
    analyzer = EmbeddingNetwork7().to(device)
    analyzer.initialize_weights()

    params = torch.FloatTensor(window_size_samples).normal_(0, 0.01).to(
        device)
    params = nn.Parameter(params)


    optisynth = OptiSynth(
        analyzer=analyzer,
        synthesizer=synth,
        synth_params=params,
        loss=nn.MSELoss(),
        n_iterations=10000)

    snd = Sound.random()
    index = np.random.randint(0, len(snd.resampled) - window_size_samples)
    samples = snd.resampled[index: index + window_size_samples]
    original_samples = \
        zounds.AudioSamples(samples, samplerate).pad_with_silence()
    original_samples /= original_samples.max()

    raw_fft = zounds.spectral.stft(original_samples)
    spectrogram = np.abs(raw_fft)
    rainbowgram = zounds.spectral.rainbowgram(raw_fft)
    samples = torch \
        .from_numpy(samples).float() \
        .to(device) \
        .view(1, 1, window_size_samples)

    with app.start_in_thread(args.port):
        optisynth(samples)

    app.start(args.port)

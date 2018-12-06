import zounds
from zounds.learn import DctTransform
from scipy.signal import gaussian
import torch
from torch.nn import functional as F


class MultiScaleLoss(object):
    def __init__(self, samplerate, scales, bands_per_segment, full_size):
        self.full_size = full_size
        self.bands_per_segment = bands_per_segment
        self.scales = scales
        self.samplerate = samplerate
        self.dct = DctTransform(use_cuda=True)

        bands = []
        for i, scale in enumerate(scales):
            start_hz = 1 if i == 0 else bands[i - 1].stop_hz
            stop_hz = samplerate.nyquist * scale
            bands.append(zounds.FrequencyBand(start_hz, stop_hz))

        barks = [
            zounds.BarkScale(b, bands_per_segment)
            for b in bands]
        banks = [
            zounds.spectral.fir_filter_bank(
                bark,
                int(full_size * scale),
                zounds.SampleRate(
                    frequency=samplerate.frequency / scale,
                    duration=samplerate.duration / scale),
                gaussian(100, 3))
            for bark, scale in zip(barks, self.scales)
            ]

        weights = [zounds.AWeighting()._wdata(b) for b in barks]
        self.weights = [
            torch.from_numpy(w).float().view(1, len(b), 1)
            for w, b in zip(weights, barks)]

        self.banks = [
            torch.from_numpy(bank).float().view(
                bands_per_segment, 1, -1).contiguous()
            for bank in banks]

    def transform_np(self, x):
        x = torch.from_numpy(x).float().cuda().view(-1, 1, 512)
        x = self._transform(x)
        return [y.data.cpu().numpy().squeeze() for y in x]

    def transform_sine(self, hz):
        synth = zounds.SineSynthesizer(self.samplerate)
        samples = synth.synthesize(
            self.samplerate.frequency * self.full_size, [hz])
        return self.transform_np(samples)

    def transform_silence(self):
        synth = zounds.SilenceSynthesizer(self.samplerate)
        samples = synth.synthesize(
            self.samplerate.frequency * self.full_size)
        return self.transform_np(samples)

    def _transform(self, x):
        if isinstance(x, list):
            bands = x
        else:
            bands = self.dct.frequency_decomposition(x, self.scales)
        features = []

        for band, weight, freq_weight in zip(bands, self.banks, self.weights):
            band = band.view(band.shape[0], 1, band.shape[-1])
            f = F.conv1d(band, weight, stride=1, padding=weight.shape[-1])
            f = F.relu(f)
            f = torch.log(1 + f * 10)
            # f = f * freq_weight
            features.append(f)
        return features

    def _distance(self, x, y):
        # spectral = -F.cosine_similarity(
        #     x.view(x.shape[0], -1), y.view(y.shape[0], -1)).mean()
        # return (((x - y) ** 2).mean() * 100) + spectral
        return ((x - y) ** 2).mean()

    def forward(self, input, target):

        input_bands = self._transform(input)
        target_bands = self._transform(target)

        # i = torch.cat(input_bands, dim=-1)
        # t = torch.cat(target_bands, dim=-1)
        # return self._distance(i, t)

        loss = None
        for i, t in zip(input_bands, target_bands):
            if loss is None:
                loss = self._distance(i, t)
            else:
                loss = loss + self._distance(i, t)
        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cuda(self, device=None):
        self.dct = self.dct.cuda()
        self.banks = [bank.cuda() for bank in self.banks]
        self.weights = [weight.cuda() for weight in self.weights]
        return self

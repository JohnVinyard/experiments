import torch
import zounds
from torch.nn import functional as F

class PerceptualFFTLoss(object):
    def __init__(self, frame_size, hop_size, samplerate):
        super(PerceptualFFTLoss, self).__init__()
        self.hop_size = hop_size
        self.frame_size = frame_size
        self.fft_size = self.frame_size // 2 + 1
        linear = zounds.LinearScale.from_sample_rate(samplerate, self.fft_size)
        weights = zounds.AWeighting()._wdata(linear)
        self.weights = torch.from_numpy(weights).float().view(1, 1, -1, 1)
        self.window = torch.hann_window(self.frame_size)

    def cuda(self, device=None):
        self.weights = self.weights.cuda(device=device)
        self.window = self.window.cuda(device=device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _transform(self, x):
        x = torch.stft(
            x,
            self.frame_size,
            self.hop_size,
            normalized=True,
            window=self.window)
        x = torch.sign(x) * torch.log(torch.abs(x * 10) + 1)
        x = x * self.weights
        return x

    def forward(self, input, target):
        i = self._transform(input.squeeze())
        t = self._transform(target.squeeze())
        return -F.cosine_similarity(i, t).mean()

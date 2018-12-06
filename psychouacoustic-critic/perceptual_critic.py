import zounds
from zounds.learn import GanExperiment
from zounds.learn import Conv1d, ConvTranspose1d, DctTransform, sample_norm
import torch
from torch import nn
from torch.nn import functional as F

latent_dim = 100
sample_size = 8192


def sample_norm(x):
    # norms = torch.norm(x, dim=1, keepdim=True)
    # return x / norms
    return x


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.activation = lambda x: F.leaky_relu(x, 0.2)
#         self.main = nn.Sequential(
#             ConvTranspose1d(
#                 latent_dim, 512, 4, 1, 0,
#                 sample_norm=True, dropout=False, activation=self.activation),
#             ConvTranspose1d(
#                 512, 512, 8, 4, 2,
#                 sample_norm=True, dropout=False, activation=self.activation),
#             ConvTranspose1d(
#                 512, 512, 8, 4, 2,
#                 sample_norm=True, dropout=False, activation=self.activation),
#             ConvTranspose1d(
#                 512, 512, 8, 4, 2,
#                 sample_norm=True, dropout=False, activation=self.activation),
#             ConvTranspose1d(
#                 512, 512, 8, 4, 2,
#                 sample_norm=True, dropout=False, activation=self.activation))
#
#         self.to_samples = ConvTranspose1d(
#             512, 1, 16, 8, 4,
#             sample_norm=False,
#             batch_norm=False,
#             dropout=False,
#             activation=None)
#
#     def forward(self, x):
#         x = x.view(-1, latent_dim, 1)
#
#         for m in self.main:
#             nx = m(x)
#             factor = nx.shape[1] // x.shape[1]
#             if nx.shape[-1] == x.shape[-1] and factor:
#                 upsampled = F.upsample(x, scale_factor=factor, mode='linear')
#                 x = self.activation(x + upsampled)
#             else:
#                 x = nx
#
#         x = self.to_samples(x)
#         return x.view(-1, sample_size)


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

        # TODO: Is overlap to blame for the noise?
        # TODO: Try a hanning window at the "synthesis" step
        # TODO: Do more weights at the final step make the problem worse,
        #   or better?
        # TODO: Consider freezing the final layer to be morlet wavelets, or
        # something
        self.final = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)
        self.gate = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)

        # n_filters = 16
        # self.dilated = nn.Sequential(
        #     nn.Conv1d(1, n_filters, 2, 1, dilation=1, padding=1, bias=False),
        #     nn.Conv1d(n_filters, n_filters, 2, 1, dilation=2, padding=1,
        #               bias=False),
        #     nn.Conv1d(n_filters, n_filters, 2, 1, dilation=4, padding=2,
        #               bias=False),
        #     nn.Conv1d(n_filters, n_filters, 2, 1, dilation=8, padding=4,
        #               bias=False),
        #     nn.Conv1d(n_filters, n_filters, 2, 1, dilation=16, padding=8,
        #               bias=False),
        #     nn.Conv1d(n_filters, n_filters, 2, 1, dilation=32, padding=16,
        #               bias=False),
        #
        # )
        #
        # self.final_final = nn.Conv1d(
        #     n_filters, 1, 2, 1, dilation=64, padding=32, bias=False)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        x = self.main(x)
        # TODO: at this point, activations in x should be fairly sparse, and
        # should ideally have weights on a logarithmic scale

        # TODO: Does a gate at this phase make sense?  Shouldn't we gate the
        # value passed to self.final
        c = self.final(x)
        g = self.gate(x)
        x = F.sigmoid(g) * c

        # for d in self.dilated:
        #     x = d(x)
        #     x = sample_norm(x)
        #     x = F.tanh(x)
        #
        # x = self.final_final(x)
        # return x[..., :sample_size].contiguous()

        return torch.clamp(x, -1, 1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.last_dim = 512
        self.activation = F.elu
        self.dct_transform = DctTransform(use_cuda=True)

        self.audio = nn.Sequential(
            nn.Conv1d(1, 512, 16, 8, 4, bias=False),
            nn.Conv1d(512, 512, 8, 4, 2, bias=False),
            nn.Conv1d(512, 512, 8, 4, 2, bias=False),
            nn.Conv1d(512, 512, 4, 2, 2, bias=False)
        )

        # TODO: consider first convolving over the dct channels
        self.spectral = Conv1d(
            512, 512, 1, 1, 1,
            batch_norm=False, dropout=False, activation=self.activation)

        self.main = nn.Sequential(
            Conv1d(
                1024, 512, 4, 2, 2,
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

        # compute features given raw audio
        audio = x
        for m in self.audio:
            audio = m(audio)
            audio = self.activation(audio)

        # TODO: try log-spaced frequencies
        # TODO: try applying log weighting to the dct coefficients

        # do an explicit frequency short-time fourier transform-type operation
        x = self.dct_transform.short_time_dct(
            x, 512, 256, zounds.HanningWindowingFunc())
        maxes, indices = torch.max(torch.abs(x), dim=1, keepdim=True)
        spectral = x / maxes
        spectral = self.spectral(spectral)
        spectral = self.activation(spectral)

        x = torch.cat([audio, spectral], dim=1)

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


if __name__ == '__main__':
    def real_sample_transformer(windowed):
        return windowed[zounds.Seconds(10):-zounds.Seconds(15)]


    experiment = GanExperiment(
        'perceptual',
        zounds.InternetArchive('AOC11B'),
        GanPair(),
        real_sample_transformer=real_sample_transformer,
        latent_dim=latent_dim,
        sample_size=sample_size,
        n_critic_iterations=10,
        n_samples=int(4e5))
    experiment.run()

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.init import xavier_normal, calculate_gain


def initialize(module, activation='leaky_relu', param=0.2, final_linear=True):
    for i, p in enumerate(module.parameters()):
        if p.data.dim() < 2:
            continue
        if i == len(module) - 1 and final_linear:
            gain = 1
        else:
            gain = calculate_gain(activation, param)
        p.data = xavier_normal(p.data, gain)


class LatentGenerator(nn.Module):
    def __init__(self, latent_dim, n_filters, out_channels):
        super(LatentGenerator, self).__init__()
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, n_filters, 2, 1, 0, bias=False),
            nn.ConvTranspose1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.ConvTranspose1d(n_filters, n_filters, 8, 2, 3, bias=False),
        )
        initialize(self.main, final_linear=False)

        self.dilations = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=32, padding=16, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=16, padding=8, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=8, padding=4, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=4, padding=2, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=2, padding=1, bias=False),
            nn.Conv1d(n_filters, out_channels, 2, 1, dilation=1, padding=1, bias=False),
        )
        initialize(self.dilations, final_linear=True)

    def forward(self, x):
        x = x.view(-1, 128, 1)

        for i, m in enumerate(self.main):
            x = m(x)
            x = F.leaky_relu(x, 0.2)

        for i, d in enumerate(self.dilations):
            nx = d(x)
            if x.shape[1] == nx.shape[1]:
                nx = nx + x
            if i < len(self.dilations) - 1:
                nx = F.leaky_relu(nx, 0.2)
            x = nx

        x = x[..., :64].contiguous()
        x = F.relu(x)
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x


class Predictor(nn.Module):
    def __init__(self, n_filters, in_channels):
        super(Predictor, self).__init__()
        self.dilations = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, 2, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=2, padding=1,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=4, padding=2,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=8, padding=4,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=16, padding=8,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=32, padding=16,
                      bias=False),
        )
        for p in self.dilations.parameters():
            p.data = xavier_normal(p.data, calculate_gain('tanh'))

        self.gates = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, 2, 1, dilation=1, padding=1,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=2, padding=1,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=4, padding=2,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=8, padding=4,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=16, padding=8,
                      bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=32, padding=16,
                      bias=False),
        )
        for p in self.dilations.parameters():
            p.data = xavier_normal(p.data, calculate_gain('sigmoid'))

        self.prediction = nn.Linear(n_filters, 128)
        for p in self.prediction.parameters():
            if p.dim() < 2:
                continue
            p.data = xavier_normal(p.data, 1)

    def forward(self, x):
        x = x.view(-1, 128, 63)

        for d, g in zip(self.dilations, self.gates):
            nx = F.tanh(d(x)) * F.sigmoid(g(x))
            if x.shape[1] == nx.shape[1]:
                nx = nx + x
            x = nx

        x = x[..., -1:]
        x = x.view(-1, self.prediction.in_features)
        x = self.prediction(x)
        return x.view(-1, 128, 1)


class LatentCritic(nn.Module):
    def __init__(self, n_filters, in_channels):
        super(LatentCritic, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters

        self.dilations = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, 2, 1, dilation=1, padding=1, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=2, padding=1, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=4, padding=2, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=8, padding=4, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=16, padding=8, bias=False),
            nn.Conv1d(n_filters, n_filters, 2, 1, dilation=32, padding=16, bias=False),
        )
        initialize(self.dilations, final_linear=False)

        self.main = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
            nn.Conv1d(n_filters, n_filters, 8, 2, 3, bias=False),
        )
        initialize(self.main, final_linear=False)

        self.final = nn.Sequential(
            nn.Linear(n_filters, n_filters // 2),
            nn.Linear(n_filters // 2, 1)
        )
        initialize(self.final, final_linear=True)

    def forward(self, x):
        x = x.view(-1, 128, 64)

        for d in self.dilations:
            nx = d(x)
            if x.shape[1] == nx.shape[1]:
                nx = nx + x
            nx = F.leaky_relu(nx, 0.2)
            x = nx

        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)

        x = x.view(-1, self.n_filters)
        for i, f in enumerate(self.final):
            x = f(x)
            if i < len(self.final) - 1:
                x = F.leaky_relu(x, 0.2)
        return x


class LatentGan(nn.Module):
    def __init__(self):
        super(LatentGan, self).__init__()
        self.generator = LatentGenerator(
            latent_dim=128, n_filters=512, out_channels=128)
        self.discriminator = LatentCritic(n_filters=512, in_channels=128)
        self.predictor = Predictor(n_filters=512, in_channels=128)


if __name__ == '__main__':
    latent_dim = 128
    out_channels = 128
    n_filters = 33

    g = LatentGenerator(latent_dim, n_filters, out_channels)
    c = LatentCritic(n_filters, out_channels)

    v = torch.FloatTensor(3, latent_dim).normal_(0, 1)

    generated = g(v)
    print 'GENERATED', generated.shape
    assessed = c(generated)
    print 'DISC', assessed.shape

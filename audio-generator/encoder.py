from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
from zounds.learn import GatedConvLayer


class Encoder(nn.Module):
    def __init__(self, latent_dim, n_filters):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        n = n_filters
        self.n = n
        self.main = nn.Sequential(
            nn.Conv1d(1, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
        )

        self.linear = nn.Linear(n, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 8192)
        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)
        x = x.view(-1, self.n)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    batch = torch.FloatTensor(3, 1, 8192)
    batch = Variable(batch)
    network = Encoder(32, 16)
    output = network(batch)
    print output.shape

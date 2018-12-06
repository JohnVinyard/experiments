import torch
from torch.autograd import Variable


class NormalDistribution(object):
    def __init__(self, latent_dim):
        super(NormalDistribution, self).__init__()
        self.latent_dim = latent_dim
        self.use_cuda = False

    def cuda(self, device=None):
        self.use_cuda = True
        return self

    def sample(self, n_samples):
        tensor_cls = \
            torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        t = tensor_cls(n_samples, self.latent_dim)
        t.normal_(0, 1)
        return t

    def sample_variable(self, n_samples):
        return Variable(self.sample(n_samples))


class UniformUnitSphere(object):
    def __init__(self, latent_dim):
        super(UniformUnitSphere, self).__init__()
        self.latent_dim = latent_dim

    def cuda(self, device=None):
        self.use_cuda = True
        return self

    def sample(self, n_samples):
        tensor_cls = \
            torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        t = tensor_cls(n_samples, self.latent_dim)
        t.uniform_(-1, 1)
        norms = torch.norm(t, p=2, dim=1, keepdim=True)
        return t / (norms + 1e-8)

    def sample_variable(self, n_samples):
        return Variable(self.sample(n_samples))


if __name__ == '__main__':
    dist = NormalDistribution(32)
    print dist.sample(20)

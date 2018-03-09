import torch


class L2NormLoss(object):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def cuda(self):
        return self

    def forward(self, input, target):
        return torch.norm(input - target, dim=-1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

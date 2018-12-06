from zounds.learn import DctTransform
from zounds.learn import CategoricalLoss


class MultiScaleCategoricalLoss(object):
    def __init__(self, scales, bits, gains):
        super(MultiScaleCategoricalLoss, self).__init__()
        self.gains = gains
        self.bits = bits
        self.scales = scales
        self.losses = [CategoricalLoss(bits) for bits in self.bits]
        self.dct = DctTransform()

    def forward(self, input, target):


        target_bands = self.dct.frequency_decomposition(target, self.scales)
        target_bands = [t * g for t, g in zip(target_bands, self.gains)]

        loss = None

        for l, ib, tb in zip(self.losses, input, target_bands):
            if loss is None:
                loss = l(ib, tb)
            else:
                loss = loss + l(ib, tb)
        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cuda(self, device=None):
        self.dct = self.dct.cuda()
        self.losses = [l.cuda() for l in self.losses]
        return self

from latent_model import LatentGan
from experiment import WithEncodings
import numpy as np
import torch
from torch.optim import Adam
import zounds
from zounds.learn import apply_network

latent_network = LatentGan()

from threading import Thread
    from time import sleep

    class SamplePool(Thread):
        def __init__(self, group=None, target=None, name=None, args=(),
                     kwargs=None, verbose=None):
            super(SamplePool, self).__init__(group, target, name, args, kwargs,
                                             verbose)
            self.samples = np.zeros((int(1e4), 64, 128), dtype=np.float32)

        def start(self):
            for snd in WithEncodings:
                x = snd.sliding_latent[:-70]
                indices = np.random.randint(0, self.samples.shape[0], x.shape[0])
                self.samples[indices] = x
                self.samples -= self.samples.mean(axis=-1, keepdims=True)
                self.samples /= (self.samples.std(axis=-1, keepdims=True) + 1e-8)
            super(SamplePool, self).start()

        def run(self):
            while True:
                snd = WithEncodings.random()
                x = snd.sliding_latent[:-70]
                print 'fetched samples with shape', x.shape
                indices = np.random.randint(0, self.samples.shape[0], x.shape[0])
                self.samples[indices] = x
                self.samples -= self.samples.mean(axis=-1, keepdims=True)
                self.samples /= (self.samples.std(axis=-1, keepdims=True) + 1e-8)
                sleep(10)

        def minibatch(self, batch_size):
            indices = np.random.randint(0, len(self.samples), batch_size)
            batch = self.samples[indices]
            batch = torch.from_numpy(batch).transpose(1, 2).contiguous().cuda()
            return batch[:, :, :63], batch[:, :, 63:]


    latent_network = latent_network.cuda()
    example_dims = WithEncodings.random().scaled.dimensions

    sample_pool = SamplePool()
    sample_pool.daemon = True
    sample_pool.start()
    optim = Adam(
        latent_network.discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

    def generate(steps=64):
        with torch.no_grad():
            data, label = sample_pool.minibatch(1)
            sequence = data
            for i in xrange(steps):
                next = latent_network.predictor(sequence[..., i:])
                sequence = torch.cat([sequence, next], dim=-1)

            sequence = sequence.data.cpu().numpy().squeeze().swapaxes(0, 1)
            decoded = apply_network(decoder, sequence, chunksize=16)
            decoded = decoded.squeeze()
            decoded = zounds.ArrayWithUnits(decoded, example_dims)
            synth = zounds.WindowedAudioSynthesizer()
            recon = synth.synthesize(decoded)
            return recon, sequence


    app = zounds.ZoundsApp(
        model=WithEncodings,
        visualization_feature=WithEncodings.latent,
        audio_feature=WithEncodings.resampled,
        globals=globals(),
        locals=locals())

    with app.start_in_thread(8888):
        while True:
            data, label = sample_pool.minibatch(32)
            predicted = latent_network.predictor(data)
            error = ((label - predicted) ** 2).mean()
            error.backward()
            print 'ERROR', float(error.item())
            optim.step()
        sample_pool.join()
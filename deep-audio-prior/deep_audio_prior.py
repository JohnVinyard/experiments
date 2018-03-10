from __future__ import print_function, division

from random import choice
from collections import defaultdict
import numpy as np
import torch
import zounds
from zounds.learn import gradients
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from itertools import product
from hashlib import md5

from sound import sound_cls
from dataset import Dataset
from loss import L2NormLoss
from upsampling import \
    LinearUpSamplingBlock, NearestNeighborUpsamplingBlock, \
    LearnedUpSamplingBlock, DctUpSamplingBlock
from generator import Generator
from plot import plot_gradients, plot_losses

latent_dim = 100
iterations = 2500
checkpoint_every = 200
kernel_sizes = [4, 8, 16, 32]

# stop training when the l2 norm falls below this threshold
error_threshold = 3.0

samplerate = zounds.SR11025()
sample_size = 8192
Sound = sound_cls(samplerate, sample_size)


class GeneratorWithFrozenNoiseVectorParameter(nn.Module):
    def __init__(self, n_examples, generator):
        super(GeneratorWithFrozenNoiseVectorParameter, self).__init__()
        t = torch.FloatTensor(n_examples, latent_dim, 1)
        t.uniform_(0, 0.1)
        self.noise = Variable(t)
        self.network = generator
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            p.data.normal_(0, 0.02)

    def cuda(self, device=None):
        self.noise = self.noise.cuda()
        super(GeneratorWithFrozenNoiseVectorParameter, self).cuda(device=device)
        return self

    def forward(self, _):
        return self.network(self.noise)[..., :sample_size]


def corrupted_segment(snd):
    """
    Choose a random segment of audio samples from snd, and corrupt the audio
    with gaussian noise
    """
    segment = choice(snd.long_windowed)
    samples = zounds.AudioSamples(segment, samplerate)
    scaled = zounds.instance_scale(samples)
    corrupted = scaled + np.random.normal(0, 0.1, samples.shape)
    return corrupted


def optimize(corrupted, network, loss, optimizer):
    """
    Perform a single optimization step
    """
    network.zero_grad()
    fake = network(None)
    err = loss(fake, corrupted)
    err.backward()
    optimizer.step()
    restored = zounds.AudioSamples(
        fake.data.cpu().numpy().squeeze(), samplerate) \
        .pad_with_silence(zounds.Seconds(1))
    return restored, err.data.cpu().numpy().squeeze()


def checkpoint_restoration(
        iteration,
        _id,
        upsampling_name,
        restored_audio,
        checkpoint_every=200):
    """
    Dump the current restored audio to disk
    """
    if iteration > 0 and iteration % checkpoint_every == 0:
        filename = 'samples/{_id}_{upsampling_name}_iter_{iteration}.wav' \
            .format(**locals())
        with open(filename, 'wb') as f:
            restored_audio.encode(f)


def main():
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.long_windowed,
        globals=globals(),
        locals=locals())

    zounds.ingest(Dataset(), Sound, multi_threaded=True)

    loss = L2NormLoss()

    upsampling_methods = [
        LearnedUpSamplingBlock,
        LinearUpSamplingBlock,
        DctUpSamplingBlock,
        NearestNeighborUpsamplingBlock
    ]

    # choose a corrupted segment from each sound
    segments = dict((snd._id, corrupted_segment(snd)) for snd in Sound)

    # create a generator over the cartesian product of corrupted segments and
    # available upsampling methods
    experiments = product(segments, upsampling_methods)

    losses = defaultdict(list)
    grads = defaultdict(lambda: defaultdict(list))

    with app.start_in_thread(8888):

        for sound_id, upsampling_method in experiments:
            upsampling_name = upsampling_method.upsampling_type
            experiment_name = '{sound_id}_{upsampling_name}'.format(**locals())
            print(experiment_name)

            _id = md5(sound_id).hexdigest()[:5]

            segment = segments[sound_id]
            segment_v = Variable(torch.from_numpy(segment).float()) \
                .view(-1, 1, sample_size).cuda()
            real_audio = segment.pad_with_silence(zounds.Seconds(1))

            with open('samples/{_id}.wav'.format(**locals()), 'wb') as f:
                real_audio.encode(f)

            # re-initialize the network
            generator = Generator(latent_dim, upsampling_method, kernel_sizes)
            network = GeneratorWithFrozenNoiseVectorParameter(
                segment_v.shape[0], generator)
            network = network.cuda()

            generator_optim = Adam(
                network.parameters(), lr=0.0001, betas=(0, 0.9))

            # optimize for a single audio segment
            for i in xrange(iterations):
                restored_audio, error = optimize(
                    segment_v, network, loss, generator_optim)

                checkpoint_restoration(
                    i,
                    _id,
                    upsampling_name,
                    restored_audio,
                    checkpoint_every=checkpoint_every)

                print('LOSS', i, error)

                # record the loss
                losses[(_id, upsampling_name)].append(error)

                # record the gradients
                for name, mn, mx, mean in gradients(network):
                    g = max(abs(mn), abs(mx))
                    bucket = grads[(_id, upsampling_name)][name].append(g)

                if error < error_threshold:
                    break

    plot_losses(losses)
    plot_gradients(grads)
    app.start(8888)


if __name__ == '__main__':
    main()

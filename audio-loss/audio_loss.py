"""
Investigate the value of a perceptual loss that approximates the early stages of
the human auditory processing pipeline.

Does an extremely low capacity neural network learn to "spend" that capacity
on more perceptually relevant/salient aspects of the signal?
"""

from __future__ import print_function

import torch
from torch import nn
from torch.optim import Adam
import hashlib

import zounds
from generator import UpsamplingGenerator
from zounds.learn import PerceptualLoss

samplerate = zounds.SR11025()
window_size = 8192
hop = 128
latent_dim = 128

wscheme = zounds.SampleRate(
    frequency=samplerate.frequency * hop,
    duration=samplerate.frequency * window_size)

BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)


@zounds.simple_lmdb_settings('loss', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        needs=BaseModel.resampled,
        wscheme=wscheme,
        wfunc=None)

    scaled = zounds.ArrayWithUnitsFeature(
        lambda x: zounds.instance_scale(x, axis=1),
        needs=windowed)


class Experiment(object):
    def __init__(self, name, target, loss, iterations, checkpoint_every):
        super(Experiment, self).__init__()
        self.checkpoint_every = checkpoint_every
        self.iterations = iterations
        self.name = name
        self.loss = loss
        self.target = target
        self.generated = None

    @property
    def real_samples(self):
        samples = zounds.instance_scale(self.target).squeeze()
        return zounds.AudioSamples(samples, samplerate).pad_with_silence()

    @property
    def real_spectral(self):
        x = self.real_samples
        return self._spectral(x)

    @property
    def fake_samples(self):
        samples = zounds.instance_scale(self.generated)
        return zounds.AudioSamples(samples, samplerate) \
            .pad_with_silence()

    @property
    def fake_spectral(self):
        x = self.fake_samples
        return self._spectral(x)

    def _audio_generation_checkpoint(self, epoch):
        checkpoint_file_name = \
            '{experiment_name}_generated_epoch{epoch}.wav' \
                .format(experiment_name=self.name, epoch=epoch)
        self.fake_samples.save(checkpoint_file_name)

    def run(self):

        real_file_name = '{experiment_name}_original.wav'\
            .format(experiment_name=self.name)
        self.real_samples.save(real_file_name)

        target = torch.from_numpy(self.target).cuda()

        network = UpsamplingGenerator(
            latent_dim=latent_dim, n_channels=32).cuda()

        optimizer = Adam(network.parameters(), lr=0.0001, betas=(0, 0.9))

        noise = torch.FloatTensor(latent_dim).normal_(0, 1).cuda()

        for i in xrange(self.iterations):
            network.zero_grad()
            samples = network(noise)
            self.generated = samples.data.cpu().numpy().squeeze()
            error = self.loss(samples, target)
            error.backward()
            optimizer.step()

            if i > 0 and i % self.checkpoint_every == 0:
                print(self.name, i, float(error.data[0]))
                self._audio_generation_checkpoint(epoch=i)

        self._audio_generation_checkpoint(epoch=i)


# TODO: Only move to cuda if it's available using device() method
# TODO: Fix warnings about initialization technique
# TODO: Introduce argparse
if __name__ == '__main__':

    urls = [
        'https://archive.org/download/AOC11B/onclassical_luisi_bach_partita_G-major_bwv-829_2.ogg',
        'https://archive.org/download/TopGunAnthem/Berlin%20-%20Take%20My%20Breath%20Away.ogg',
        'https://ia802708.us.archive.org/20/items/LucaBrasi2/06-Kevin_Gates-Out_The_Mud_Prod_By_The_Runners_The_Monarch.ogg',
        'https://archive.org/download/Greatest_Speeches_of_the_20th_Century/CheckersSpeech.ogg'
    ]

    iterations = 1000
    checkpoint_every = 250

    for url in urls:
        if not Sound.exists(url):
            print('processing {url}'.format(**locals()))
            Sound.process(meta=url, _id=url)
        else:
            print('already processed {url}'.format(**locals()))

    scale = zounds.BarkScale(
        frequency_band=zounds.FrequencyBand(1, samplerate.nyquist),
        n_bands=512)

    perceptual_loss = PerceptualLoss(
        scale,
        samplerate,
        lap=1,
        log_factor=10,
        basis_size=512,
        frequency_weighting=zounds.AWeighting(),
        cosine_similarity=True).cuda()
    mse_loss = nn.MSELoss().cuda()

    for snd in Sound:
        snd_id = hashlib.md5(snd._id).hexdigest()[:5]

        sample = snd.scaled[len(snd.scaled) // 2].reshape((1, 1, window_size))
        pl = Experiment(
            '{snd_id}_perceptual'.format(**locals()),
            sample,
            perceptual_loss,
            iterations,
            checkpoint_every)

        pl.run()
        mse = Experiment(
            '{snd_id}_mse'.format(**locals()),
            sample,
            mse_loss,
            iterations,
            checkpoint_every)
        mse.run()



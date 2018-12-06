import argparse
import zounds
import numpy as np
from zounds.learn import try_network, to_var, from_var
from torch.autograd import Variable
import torch

from autoencoder import train_autoencoder, AutoEncoder
from gan import train_gan, GanPair

samplerate = zounds.SR11025()
window_size = 512
autoencoder_latent_dim = 32
gan_latent_dim = 128
gan_sample_size = 128

window_samplerate = samplerate * (window_size // 2, window_size)

BaseModel = zounds.resampled(
    resample_to=samplerate,
    store_resampled=True)

long_window_sample_rate = zounds.SampleRate(
    frequency=window_samplerate.frequency * 1,
    duration=window_samplerate.frequency * gan_sample_size)


@zounds.simple_lmdb_settings('ae', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    scaled = zounds.AudioSamplesFeature(
        lambda x: np.clip(zounds.instance_scale(x), -1, 1),
        needs=BaseModel.resampled)

    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_samplerate,
        wfunc=None,
        needs=scaled)


def ingest_data(archive_id):
    dataset = zounds.InternetArchive(archive_id)
    zounds.ingest(dataset, Sound, multi_threaded=True)
    return dataset


# TODO: Some kind of convenience method for making assertions about network
# input and output shapes
def check_autoencoder(network):
    fake_batch = np.random.random_sample((3, 1, window_size)).astype(np.float32)
    encoded = try_network(network.encoder, fake_batch)
    print 'Encoded', encoded.shape
    assert encoded.shape == (3, autoencoder_latent_dim)
    decoded = try_network(network.decoder, from_var(encoded))
    print 'Decoded', decoded.shape
    assert decoded.shape == (3, 256, window_size)


def check_gan(network):
    fake_batch = np.random.normal(0, 1, (3, gan_latent_dim)).astype(np.float32)
    samples = try_network(network.generator, fake_batch)
    print 'Generated', samples.shape
    assert samples.shape == (3, autoencoder_latent_dim, gan_sample_size)
    wasserstein_distances = try_network(
        network.discriminator, from_var(samples))
    print 'Wasserstein Distance', wasserstein_distances.shape
    assert wasserstein_distances.shape == (3, 1)


def main(archive_id, args, sound_cls, sr):
    autoencoder = AutoEncoder(autoencoder_latent_dim, window_size)
    gan_pair = GanPair(autoencoder_latent_dim, gan_latent_dim, gan_sample_size)

    check_autoencoder(autoencoder)
    check_gan(gan_pair)

    dataset = ingest_data(archive_id)
    snd = Sound.random()

    example_dims = snd.windowed.dimensions

    def from_latent_space(z):
        """
        Transform a single generated example in the latent space to an
        AudioSamples instance
        """
        z = z.T.reshape((-1, autoencoder_latent_dim, 1))
        z = Variable(torch.from_numpy(z).float()).cuda()
        decoded = autoencoder.decoder(z)
        decoded = from_var(decoded)
        samples = zounds.inverse_one_hot(decoded, axis=1)
        samples = zounds.inverse_mu_law(samples)
        samples = zounds.ArrayWithUnits(samples, example_dims)
        samples *= np.hanning(window_size)
        synth = zounds.WindowedAudioSynthesizer()
        samples = synth.synthesize(samples)
        return samples

    ae_pipeline = train_autoencoder(
        args=args,
        sound_cls=sound_cls,
        latent_dim=autoencoder_latent_dim,
        window_size=window_size,
        samplerate=sr,
        epochs=200,
        checkpoint_epochs=10)

    # TODO: Ensure that features are re-computed when network is re-trained
    class WithEncoded(Sound):
        encoded = zounds.ArrayWithUnitsFeature(
            zounds.Learned,
            learned=ae_pipeline,
            needs=Sound.windowed,
            dtype=np.float32,
            store=True)

        long_windowed = zounds.ArrayWithUnitsFeature(
            zounds.SlidingWindow,
            wscheme=long_window_sample_rate,
            wfunc=None,
            needs=encoded)

    for snd in WithEncoded:
        print snd.long_windowed.shape

    autoencoder = ae_pipeline.pipeline[-1].network
    autoencoder.eval()

    train_gan(
        args=args,
        sound_cls=WithEncoded,
        sound_feature=WithEncoded.long_windowed,
        autoencoder_latent_dim=autoencoder_latent_dim,
        dataset=dataset,
        latent_dim=gan_latent_dim,
        sample_size=gan_sample_size,
        from_latent_space=from_latent_space,
        epochs=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.ObjectStorageSettings(),
        zounds.AppSettings()
    ])
    parser.add_argument(
        '--force-train-autoencoder',
        help='Train the autoencoder, even if weights for it already exist',
        action='store_true',
        default=False)
    parser.add_argument(
        '--force-train-gan',
        help='Train the GAN, even if weights already exist',
        action='store_true',
        default=False)

    parsed_args = parser.parse_args()
    main('AOC11B', parsed_args, Sound, samplerate)

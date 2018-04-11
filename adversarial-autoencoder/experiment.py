import numpy as np
import zounds
from model import Network
import argparse
import featureflow as ff
from trainer import AdversarialAutoencoderTrainer
from latent import NormalDistribution
from random import choice
from zounds.learn import apply_network, PerceptualLoss

experiment_name = 'adversarial-autoencoder'
samplerate = zounds.SR11025()
window_size = 512
n_filters = 64
hop_size = window_size // 2
latent_dim = 32
latent_distribution = NormalDistribution(latent_dim)
batch_size = 4

wscheme = zounds.SampleRate(
    frequency=samplerate.frequency * hop_size,
    duration=samplerate.frequency * window_size)

BaseModel = zounds.windowed(
    wscheme=wscheme,
    resample_to=samplerate,
    store_resampled=True)


def scaled(x):
    x = zounds.instance_scale(x, axis=0)
    return x


@zounds.simple_lmdb_settings(
    'adversarial', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    scaled = zounds.ArrayWithUnitsFeature(
        scaled,
        needs=BaseModel.windowed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.AppSettings(),
        zounds.ObjectStorageSettings()
    ])
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=True)

    args = parser.parse_args()

    zounds.ingest(zounds.InternetArchive('AOC11B'), Sound, multi_threaded=True)


    @zounds.object_store_pipeline_settings(
        experiment_name,
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class Pipeline(ff.BaseModel):
        wgan = ff.PickleFeature(
            zounds.PyTorchGan,
            trainer=ff.Var('trainer'))


    scale = zounds.MelScale(
        frequency_band=zounds.FrequencyBand(20, samplerate.nyquist - 300),
        n_bands=500)

    perceptual_model = PerceptualLoss(
        scale,
        samplerate,
        basis_size=512,
        lap=1,
        log_factor=10,
        frequency_weighting=zounds.AWeighting(),
        phase_locking_cutoff_hz=1200)

    try:
        network = Pipeline.load_network()
    except RuntimeError:
        network = Network(
            latent_dim,
            n_filters,
            window_size,
            perceptual_model)
        for p in network.parameters():
            p.data.normal_(0, 0.02)

    trainer = AdversarialAutoencoderTrainer(
        network=network,
        distribution=latent_distribution,
        epochs=100,
        batch_size=batch_size,
        loss=perceptual_model,
        n_critic_iterations=5,
        checkpoint_epochs=1)


    def fake_audio():
        samples = choice(trainer.generated_samples)
        return zounds.AudioSamples(samples, samplerate).pad_with_silence()


    def fake_samples():
        return np.array(fake_audio())[:window_size]


    def reconstructed_audio():
        snd = Sound.random()
        duration = zounds.Seconds(10)
        samples = snd.resampled[:duration]
        samples = zounds.instance_scale(samples)
        sr = zounds.SampleRate(
            frequency=samplerate.frequency * window_size,
            duration=samplerate.frequency * window_size)
        windowed = samples.sliding_window(sr)

        encoded = apply_network(network.encoder, windowed, chunksize=4)
        decoded = apply_network(network.generator, encoded, chunksize=4)
        decoded = decoded.squeeze()
        decoded = zounds.ArrayWithUnits.from_example(decoded, windowed)
        synth = zounds.WindowedAudioSynthesizer()
        recon = synth.synthesize(decoded)
        return snd.resampled[:duration], encoded, recon


    def latent_space():
        return trainer.latent_samples


    if args.cuda:
        perceptual_model.cuda()
        trainer.cuda()

    app = zounds.TrainingMonitorApp(
        trainer=trainer,
        keys_to_graph=(
            'data_critic_loss',
            'latent_critic_loss',
            'generator_loss',
            'encoder_loss',
            'autoencoder_loss'),
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        model=Sound,
        globals=globals(),
        locals=locals())

    with app.start_in_thread(args.port):
        Pipeline.process(
            dataset=(Sound, Sound.scaled),
            trainer=trainer,
            dtype=np.float32,
            nsamples=int(1e5))

    app.start(args.port)

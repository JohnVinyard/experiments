import argparse
from random import choice

import torch
import featureflow as ff
import numpy as np
import zounds
from zounds.learn import \
    PerceptualLoss, apply_network, model_hash, DctTransform
from latent import NormalDistribution
from latent_model import LatentGan
from model import Network
from trainer import AdversarialAutoencoderTrainer


experiment_name = 'adversarial-autoencoder'
samplerate = zounds.SR11025()
window_size = 4096
hop_size = window_size // 2

n_filters = 64
latent_dim = 16 * 6 * 5
latent_distribution = NormalDistribution(128)
batch_size = 8

feature_filter = lambda x: x[:-zounds.Seconds(14)]

wscheme = zounds.SampleRate(
    frequency=samplerate.frequency * hop_size,
    duration=samplerate.frequency * window_size)

BaseModel = zounds.windowed(
    wscheme=wscheme,
    resample_to=samplerate,
    store_resampled=True)


def scaled(x):
    x = zounds.instance_scale(x, axis=None) * 10
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

    args = parser.parse_args()

    zounds.ingest(
        # zounds.InternetArchive('AOC11B'),
        zounds.InternetArchive('CastlevaniaIIIDraculasCurseNESMusicEnterNameEpitaph'),
        # zounds.InternetArchive('LucaBrasi2'),
        # zounds.InternetArchive('TopGunAnthem'),
        # zounds.PhatDrumLoops(),
        # zounds.MusicNet('/home/user/Downloads/'),
        Sound,
        multi_threaded=True)

    print 'ingested audio'

    dct = DctTransform()

    maxes = [0] * 5
    for i, snd in enumerate(Sound):
        bands = dct.frequency_decomposition(
            torch.from_numpy(snd.scaled).float(),
            [1.0, 0.5, 0.25, 0.125, 0.0625])
        maxes = [m + float(b.std()) for m, b in zip(maxes, bands)]
    maxes = [float(x) for x in list(np.array(maxes) / float(i))]
    print maxes




    # @zounds.object_store_pipeline_settings(
    #     experiment_name,
    #     args.object_storage_region,
    #     args.object_storage_username,
    #     args.object_storage_api_key)

    @zounds.simple_lmdb_settings('autoencoder')
    @zounds.infinite_streaming_learning_pipeline
    class Pipeline(ff.BaseModel):
        wgan = ff.PickleFeature(
            zounds.PyTorchGan,
            trainer=ff.Var('trainer'))


    try:
        network = Pipeline.load_network()
        print 'loaded pre-trained network'
    except RuntimeError:
        network = Network(
            latent_dim,
            n_filters,
            window_size,
            gains=maxes)
        try:
            network.load_state_dict(torch.load('weights.dat'))
        except IOError:
            pass
        print 'created network'

    scale = zounds.BarkScale(
        zounds.FrequencyBand(1, samplerate.nyquist), 256)

    loss = PerceptualLoss(
        scale,
        samplerate,
        lap=1,
        log_factor=10,
        basis_size=256,
        frequency_weighting=zounds.AWeighting(),
        cosine_similarity=False).cuda()

    network.data_critic.perceptual_model = loss

    trainer = AdversarialAutoencoderTrainer(
        network=network,
        distribution=latent_distribution,
        epochs=100,
        batch_size=batch_size,
        autoencoder_loss=loss,
        n_critic_iterations=5,
        checkpoint_epochs=1)


    def fake_audio():
        samples = choice(trainer.generated_samples)
        return zounds.AudioSamples(samples, samplerate).pad_with_silence()


    def fake_samples():
        return np.array(fake_audio())[:window_size]


    def log_stft(x):
        return zounds.log_modulus(np.abs(zounds.spectral.stft(x)) * 100)


    def reconstructed_audio():
        snd = Sound.random()
        total_duration_seconds = \
            int(snd.resampled.dimensions[0].end / zounds.Seconds(1))
        try:
            start_seconds = np.random.randint(0, total_duration_seconds - 10)
        except ValueError:
            start_seconds = 0
        start = zounds.Seconds(start_seconds)
        end = start + zounds.Seconds(10)

        windowed = snd.scaled[start:end].astype(np.float32)

        encoded = apply_network(network.encoder, windowed, chunksize=32)
        decoded = apply_network(network.generator, encoded, chunksize=32)
        decoded = decoded.squeeze()

        decoded = zounds.ArrayWithUnits.from_example(decoded, windowed) * zounds.OggVorbisWindowingFunc()
        synth = zounds.WindowedAudioSynthesizer()
        recon = synth.synthesize(decoded)

        # here, encoded will be (batch, 30, 16), but we'd like to view all the
        # time dimensions together, so we need to:
        if encoded.ndim == 3:
            encoded = encoded.swapaxes(1, 2)
            encoded.reshape(-1, 30)

        # now we have (batch, 16, 30)
        return \
            snd.resampled[start:end], \
            encoded.squeeze(), \
            recon


    def latent_space():
        return trainer.latent_samples


    trainer.cuda()

    reconstructed_audio()

    app = zounds.TrainingMonitorApp(
        trainer=trainer,
        keys_to_graph=(
            'data_critic_loss',
            'latent_critic_loss',
            'generator_loss',
            'encoder_loss',
            'autoencoder_loss',
            'predictor_loss'),
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        model=Sound,
        secret=args.app_secret,
        globals=globals(),
        locals=locals())

    with app.start_in_thread(args.port):
        print 'started app on', args.port
        Pipeline.process(
            dataset=(Sound, Sound.scaled),
            trainer=trainer,
            dtype=np.float32,
            nsamples=int(200000),
            feature_filter=feature_filter,
            parallel=True)

    # pipeline = Pipeline()
    # encoder = pipeline.pipeline[-1].network.encoder
    # decoder = pipeline.pipeline[-1].network.generator

    encoder = network.encoder
    decoder = network.generator

    def encode(x):
        td = x.dimensions[0]
        x = apply_network(encoder, x, chunksize=16)
        return zounds.ArrayWithUnits(x, [td, zounds.IdentityDimension()])


    latent_wscheme = zounds.SampleRate(
        frequency=wscheme.frequency,
        duration=wscheme.duration * 32)


    class WithEncodings(Sound):
        latent = zounds.ArrayWithUnitsFeature(
            encode,
            needs=Sound.scaled,
            store=True,
            closure_fingerprint=lambda d: model_hash(d['encoder']))

        sliding_latent = zounds.ArrayWithUnitsFeature(
            zounds.SlidingWindow,
            wscheme=latent_wscheme,
            wfunc=None,
            needs=latent,
            store=False)



    @zounds.object_store_pipeline_settings(
        experiment_name + '_stage2',
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class LatentPipeline(ff.BaseModel):
        wgan = ff.PickleFeature(
            zounds.PyTorchGan,
            trainer=ff.Var('trainer'))


    try:
        latent_network = LatentPipeline.load_network()
        print 'loaded pre-trained latent network'
    except RuntimeError:
        latent_network = LatentGan()
        print 'created and initialized latent network'



    # means = np.concatenate(
    #     [snd.latent.mean(axis=0, keepdims=True) for snd in WithEncodings], axis=0)
    # stds = np.concatenate(
    #     [snd.latent.std(axis=0, keepdims=True) for snd in WithEncodings], axis=0)
    #
    # means = means.mean(axis=0)
    # stds = stds.mean(axis=0)
    #
    # means = torch.from_numpy(means).cuda().view(1, 128, 1)
    # stds = torch.from_numpy(stds).cuda().view(1, 128, 1)


    def preprocess_minibatch(epoch, x):
        if x.shape[1] == 64:
            x = x.transpose(1, 2).contiguous()

            # x = x - means
            # x = x / stds

            return x
            # # here, x is (batch, 64, 16, 30), so we'll
            # x = x.transpose(3, 2)
            # # and now we have (batch, 64, 30, 16), so we still need to
            # x = x.transpose(1, 2).contiguous()
            # # so now we have (batch, 30, 64, 16), and finally, we just flatten
            # # the final two time dimensions
            # x = x.view(-1, 30, 64, 16)
            #
            #
            # return x.view(-1, 30, 1024)
        else:
            return x


    latent_trainer = zounds.WassersteinGanTrainer(
        network=latent_network,
        latent_dimension=(128,),
        n_critic_iterations=5,
        epochs=1000,
        batch_size=16,
        checkpoint_epochs=1,
        preprocess_minibatch=preprocess_minibatch)

    example_dims = WithEncodings.random().scaled.dimensions


    def check_synthesis(add_noise=0):
        snd = WithEncodings.random()
        features = snd.sliding_latent[0].astype(np.float32)
        # here, features are (batch, 16, 30), but our synthesizer wants the time
        # dimension last, so we'll need to:
        # features = features.swapaxes(1, 2)
        # now, we should have (batch, 30, 16)

        if add_noise > 0:
            features += np.random.normal(0, add_noise, features.shape)

        decoded = apply_network(
            decoder, features, chunksize=32)
        decoded = decoded.squeeze()

        decoded = zounds.ArrayWithUnits(decoded, example_dims) * zounds.OggVorbisWindowingFunc()
        synth = zounds.WindowedAudioSynthesizer()
        recon = synth.synthesize(decoded)
        # features is now (batch, 30, 16), but we'd like to view the time
        # dimensions together, so:
        # return recon, features.swapaxes(1, 2).reshape((-1, 30))

        return recon, features


    def generate_latent():
        import torch
        latent = torch.FloatTensor(1, 128, 1).normal_(0, 1).cuda()

        generated = latent_network.generator(latent)

        # generated = generated * stds
        # generated = generated + means

        generated = generated.transpose(1, 2).contiguous()

        # generated = generated.view(1, 30, 64, 16)


        # here we'll have (1, 30, 1024), but the decoder will want
        # (batch, 30, 16).  What we _really_ have is (1, 30, 64, 16), so we
        # need to:
        # generated = generated.transpose(0, 2).contiguous().squeeze()
        # print generated.shape

        # here we should now have (batch, 30, 16)

        try:
            decoded = apply_network(
                decoder, generated.data.cpu().numpy(), chunksize=32)
        except:
            return None, None
        decoded = decoded.squeeze()
        decoded = zounds.ArrayWithUnits(decoded, example_dims) * zounds.OggVorbisWindowingFunc()
        synth = zounds.WindowedAudioSynthesizer()
        recon = synth.synthesize(decoded)


        generated = generated.data.cpu().numpy()
        # generated is (batch, 30, 16), but we'd like to view the time dim
        # together, so:
        # generated = generated.swapaxes(1, 2).reshape(-1, 30)

        return recon, generated.squeeze()


    app = zounds.GanTrainingMonitorApp(
        trainer=trainer,
        model=WithEncodings,
        visualization_feature=WithEncodings.scaled,
        audio_feature=WithEncodings.resampled,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    latent_trainer.cuda()

    with app.start_in_thread(args.port):
        print 'started app on', args.port
        LatentPipeline.process(
            dataset=(WithEncodings, WithEncodings.sliding_latent),
            feature_filter=lambda x: x[:-70],
            parallel=True,
            trainer=latent_trainer,
            dtype=np.float32,
            nsamples=int(1e4))

    app.start(args.port)

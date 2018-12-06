import argparse
from itertools import chain

import featureflow as ff
import numpy as np
import zounds
from torch import nn
from torch.optim import Adam
from zounds.learn import \
    from_var, model_hash, apply_network, PerceptualLoss, BandLoss
from model import AutoEncoder, Encoder
from loss import LearnedPatchLoss, BandLoss2, BandLoss3

samplerate = zounds.SR11025()
window_size = 2048
latent_dim = 128
n_filters = 64
kernel_sizes = [3, 7, 15, 31]
batch_size = 12

BaseModel = zounds.windowed(
    wscheme=samplerate * (16, window_size),
    resample_to=samplerate)


@zounds.simple_lmdb_settings('ae', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    pass


def reconstruct():
    snd = Sound.random()

    # sliding window
    _, windowed = snd.resampled.sliding_window_with_leftovers(
        window_size, window_size // 2, dopad=True)

    maxes, windowed = zounds.instance_scale(
        windowed, axis=-1, return_maxes=True)

    # encode
    encoded = apply_network(network.encoder, windowed, chunksize=64)

    # decode
    decoded = apply_network(network.decoder, encoded, chunksize=64)
    decoded = decoded.squeeze()

    decoded *= maxes
    decoded *= np.hanning(window_size)
    decoded = zounds.ArrayWithUnits.from_example(decoded, windowed)

    synth = zounds.WindowedAudioSynthesizer()
    recon = synth.synthesize(decoded)
    return encoded, recon


if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[
        zounds.ObjectStorageSettings(),
        zounds.AppSettings()
    ])
    parser.add_argument(
        '--reconstruct',
        help='reconstruct a random piece of audio using the current network',
        action='store_true',
        default=False,
        required=False)
    parser.add_argument(
        '--loss',
        help='which loss to use: (perceptual|mse|categorical)',
        default='mse')
    args = parser.parse_args()

    zounds.ingest(
        zounds.InternetArchive('LucaBrasi2'),
        Sound,
        multi_threaded=True)


    @zounds.object_store_pipeline_settings(
        'RawSampleAutoEncoder-{loss}'.format(loss=args.loss),
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class RawSampleAutoEncoderPipeline(ff.BaseModel):
        scaled = ff.Feature(
            zounds.InstanceScaling)

        autoencoder = ff.Feature(
            zounds.PyTorchAutoEncoder,
            trainer=ff.Var('trainer'),
            needs=scaled)


    try:
        network = RawSampleAutoEncoderPipeline.load_network()
        print 'loaded network with hash', model_hash(network)
    except RuntimeError as e:
        network = AutoEncoder(latent_dim, window_size, kernel_sizes, n_filters)
        for p in network.parameters():
            if p.data.dim() == 3:
                p.data.normal_(0, 2. / p.data.shape[1])
            elif p.data.dim() == 2:
                p.data.normal_(0, 1. / p.data.shape[1])
            else:
                p.data.fill_(0)

    if args.loss == 'mse':
        loss = nn.MSELoss()
    elif args.loss == 'perceptual':
        scale = zounds.BarkScale(
            zounds.FrequencyBand(20, samplerate.nyquist - 300), 500)
        loss = PerceptualLoss(
            scale,
            samplerate,
            lap=1,
            log_factor=10,
            frequency_weighting=zounds.AWeighting(),
            phase_locking_cutoff_hz=1200)
    elif args.loss == 'band':
        # loss = BandLoss(
        #     [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        #     spectral_shape_weight=0.001)
        loss = BandLoss2(
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    elif args.loss == 'learned':
        loss = LearnedPatchLoss(
            Encoder(latent_dim, window_size, kernel_sizes, n_filters))
    else:
        raise ValueError('loss {loss} not supported'.format(**locals()))

    trainer = zounds.SupervisedTrainer(
        model=network,
        loss=loss,
        optimizer=lambda model: Adam(
            chain(model.encoder.parameters(), model.decoder.parameters()),
            lr=0.0001, betas=(0, 0.9)),
        checkpoint_epochs=200,
        epochs=200,
        batch_size=batch_size,
        holdout_percent=0.25)

    trainer.cuda()
    loss.cuda()

    app = zounds.SupervisedTrainingMonitorApp(
        trainer=trainer,
        model=Sound,
        visualization_feature=Sound.windowed,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    if args.reconstruct:
        encoded, recon = reconstruct()
    else:
        with app.start_in_thread(8888):

            def example():
                inp, label = trainer.random_sample()
                inp = from_var(inp).squeeze()
                label = from_var(label).squeeze()
                inp = zounds.AudioSamples(inp, samplerate) \
                    .pad_with_silence(zounds.Seconds(1))
                label = zounds.AudioSamples(label, samplerate) \
                    .pad_with_silence(zounds.Seconds(1))
                return inp, label


            RawSampleAutoEncoderPipeline.process(
                dataset=(Sound, Sound.windowed),
                nsamples=int(1e5),
                dtype=np.float32,
                trainer=trainer,
                feature_filter=lambda x: x)

    app.start(8888)

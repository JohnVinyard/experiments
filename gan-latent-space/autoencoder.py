from torch import nn
import zounds
import featureflow as ff
from torch.optim import Adam
from zounds.learn import \
    CategoricalLoss, from_var, MultiResolutionConvLayer, apply_network
import numpy as np
import torch
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        filters = 32
        kernels = [4, 8, 16, 32]
        nk = len(kernels)
        ts = nk * filters
        self.ts = ts
        self.main = nn.Sequential(
            MultiResolutionConvLayer(1, filters, kernels, stride=4),
            MultiResolutionConvLayer(ts, filters, kernels, stride=4),
            MultiResolutionConvLayer(ts, filters, kernels, stride=4),
            MultiResolutionConvLayer(ts, filters, kernels, stride=4),
            MultiResolutionConvLayer(ts, filters, kernels, stride=4)
        )
        self.linear = nn.Linear(ts, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.window_size)
        for m in self.main:
            x = torch.cat(m(x), dim=1)
        x = x.view(-1, self.ts)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        filters = 32
        kernels = [4, 8, 16, 32]
        nk = len(kernels)
        ts = nk * filters
        self.ts = ts
        self.upsamplers = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim, 4, 3, bias=False),
            nn.ConvTranspose1d(ts, ts, 4, 3, bias=False),
            nn.ConvTranspose1d(ts, ts, 4, 3, bias=False),
            nn.ConvTranspose1d(ts, ts, 4, 3, bias=False),
            nn.ConvTranspose1d(ts, ts, 4, 4, bias=False),
        )

        self.main = nn.Sequential(
            MultiResolutionConvLayer(latent_dim, filters, kernels),
            MultiResolutionConvLayer(ts, filters, kernels),
            MultiResolutionConvLayer(ts, filters, kernels),
            MultiResolutionConvLayer(ts, filters, kernels),
            MultiResolutionConvLayer(ts, filters, kernels),
        )
        self.final = nn.Conv1d(ts, 256, 16, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        for upsample, m in zip(self.upsamplers, self.main):
            x = upsample(x)
            x = torch.cat(m(x), dim=1)
        x = self.final(x)
        x = F.log_softmax(x, dim=1)
        x = x[..., :self.window_size].contiguous()
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.encoder = Encoder(latent_dim, window_size)
        self.decoder = Decoder(latent_dim, window_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
        args,
        sound_cls,
        latent_dim,
        window_size,
        samplerate,
        epochs=100,
        checkpoint_epochs=1):
    Sound = sound_cls

    @zounds.object_store_pipeline_settings(
        'GanLatentSpaceAutoEncoder',
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class RawSampleAutoEncoderPipeline(ff.BaseModel):
        autoencoder = ff.Feature(
            zounds.PyTorchAutoEncoder,
            trainer=ff.Var('trainer'))

    exists = RawSampleAutoEncoderPipeline.exists()
    force = args.force_train_autoencoder

    if not exists or force:
        # TODO: weight initialization methods in zounds
        try:
            # TODO: Loading the network from pre-existing weights is just
            # another weight initialization scheme
            network = RawSampleAutoEncoderPipeline.load_network()
        except RuntimeError:
            network = AutoEncoder(latent_dim, window_size)
            for p in network.parameters():
                p.data.normal_(0, 0.02)
    else:
        return RawSampleAutoEncoderPipeline()

    trainer = zounds.SupervisedTrainer(
        model=network,
        loss=CategoricalLoss(255),
        optimizer=lambda model: Adam(model.parameters(), lr=0.00005),
        checkpoint_epochs=checkpoint_epochs,
        epochs=epochs,
        batch_size=32,
        holdout_percent=0.25)

    def example():
        inp, label = trainer.random_sample()
        inp = from_var(inp).squeeze()
        label = from_var(label).squeeze()
        inp = zounds.AudioSamples(inp, samplerate) \
            .pad_with_silence(zounds.Seconds(1))
        label = zounds.inverse_one_hot(label, axis=0)
        label = zounds.inverse_mu_law(label)
        label = zounds.AudioSamples(label, samplerate) \
            .pad_with_silence(zounds.Seconds(1))
        return inp, label

    def reconstruct():
        snd = Sound.random()
        windowed = snd.windowed[:zounds.Seconds(10)]
        encoded = apply_network(network.encoder, windowed, chunksize=16)
        decoded = apply_network(network.decoder, encoded, chunksize=16)
        decoded = zounds.inverse_one_hot(decoded, axis=1)
        decoded = zounds.inverse_mu_law(decoded)
        decoded *= np.hanning(window_size)
        decoded = zounds.ArrayWithUnits.from_example(decoded, windowed)
        synth = zounds.WindowedAudioSynthesizer()
        recon = synth.synthesize(decoded)
        return encoded, recon

    app = zounds.SupervisedTrainingMonitorApp(
        trainer=trainer,
        model=Sound,
        visualization_feature=Sound.windowed,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    try:
        with app.start_in_thread(8888):
            RawSampleAutoEncoderPipeline.process(
                dataset=(Sound, Sound.windowed),
                nsamples=int(1e5),
                dtype=np.float32,
                trainer=trainer)
    except KeyboardInterrupt:
        print 'Suspended autoencoder training...'
        pass

    return RawSampleAutoEncoderPipeline()

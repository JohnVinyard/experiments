from torch import nn
from zounds.learn import MultiResolutionConvLayer, from_var
import torch
import zounds
import featureflow as ff
import numpy as np
from random import choice


class Generator(nn.Module):
    def __init__(self, autoencoder_latent_dim, gan_latent_dim, gan_sample_size):
        super(Generator, self).__init__()
        self.gan_sample_size = gan_sample_size
        self.gan_latent_dim = gan_latent_dim
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.n_filters = 64
        self.kernels = [2, 4, 8]
        self.ts = len(self.kernels) * self.n_filters

        self.upsamplers = nn.Sequential(
            nn.ConvTranspose1d(
                self.gan_latent_dim, self.gan_latent_dim, 4, 2, bias=False),
            nn.ConvTranspose1d(self.ts, self.ts, 4, 2, bias=False),
            nn.ConvTranspose1d(self.ts, self.ts, 4, 2, bias=False),
            nn.ConvTranspose1d(self.ts, self.ts, 4, 2, bias=False),
            nn.ConvTranspose1d(self.ts, self.ts, 4, 2, bias=False),
        )

        self.main = nn.Sequential(
            MultiResolutionConvLayer(
                self.gan_latent_dim, self.n_filters, self.kernels),
            MultiResolutionConvLayer(self.ts, self.n_filters, self.kernels),
            MultiResolutionConvLayer(self.ts, self.n_filters, self.kernels),
            MultiResolutionConvLayer(self.ts, self.n_filters, self.kernels),
            MultiResolutionConvLayer(self.ts, self.n_filters, self.kernels)
        )

        self.final = nn.Conv1d(
            self.ts, self.autoencoder_latent_dim, 16, 1, padding=9)

    def forward(self, x):
        x = x.view(-1, self.gan_latent_dim, 1)
        for upsample, m in zip(self.upsamplers, self.main):
            x = upsample(x)
            x = torch.cat(m(x), dim=1)
        x = self.final(x)
        return x


class Critic(nn.Module):
    def __init__(self, autoencoder_latent_dim, gan_sample_size):
        super(Critic, self).__init__()
        self.gan_sample_size = gan_sample_size
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.n_filters = 64
        self.kernels = [2, 4, 8]
        self.ts = len(self.kernels) * self.n_filters
        self.main = nn.Sequential(
            MultiResolutionConvLayer(
                self.autoencoder_latent_dim,
                self.n_filters,
                self.kernels,
                stride=4),
            MultiResolutionConvLayer(
                self.ts, self.n_filters, self.kernels, stride=4),
            MultiResolutionConvLayer(
                self.ts, self.n_filters, self.kernels, stride=4),
            MultiResolutionConvLayer(
                self.ts, self.n_filters, self.kernels, stride=4),
            MultiResolutionConvLayer(
                self.ts, self.n_filters, self.kernels, stride=2)
        )
        self.final = nn.Linear(self.ts, 1)

    def forward(self, x):
        if x.shape[1] != self.autoencoder_latent_dim:
            x = x.transpose(1, 2).contiguous()

        x = x.view(-1, self.autoencoder_latent_dim, self.gan_sample_size)
        for m in self.main:
            x = torch.cat(m(x), dim=1)
        x = x.squeeze()
        x = self.final(x)
        return x


class GanPair(nn.Module):
    def __init__(self, autoencoder_latent_dim, gan_latent_dim, sample_size):
        self.autoencoder_latent_dim = autoencoder_latent_dim
        self.gan_latent_dim = gan_latent_dim
        self.sample_size = sample_size
        super(GanPair, self).__init__()
        self.generator = Generator(
            autoencoder_latent_dim, gan_latent_dim, sample_size)
        self.discriminator = Critic(autoencoder_latent_dim, sample_size)

    def forward(self, x):
        raise NotImplementedError()


def train_gan(
        args,
        sound_cls,
        sound_feature,
        autoencoder_latent_dim,
        dataset,
        latent_dim,
        sample_size,
        from_latent_space,
        epochs=100):

    @zounds.object_store_pipeline_settings(
        'Gan-latent-space',
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class Gan(ff.BaseModel):
        wgan = ff.PickleFeature(
            zounds.PyTorchGan,
            trainer=ff.Var('trainer'))

    exists = Gan.exists()
    force = args.force_train_gan

    if not exists or force:
        try:
            network = Gan.load_network()
        except RuntimeError:
            network = GanPair(
                autoencoder_latent_dim=autoencoder_latent_dim,
                gan_latent_dim=latent_dim,
                sample_size=sample_size)
            for p in network.parameters():
                p.data.normal_(0, 0.02)
    else:
        return Gan()

    trainer = zounds.WassersteinGanTrainer(
        network,
        latent_dimension=(latent_dim,),
        n_critic_iterations=5,
        epochs=epochs,
        batch_size=16)

    samples = [None]

    def batch_complete(*args, **kwargs):
        samples[0] = from_var(kwargs['samples']).squeeze()

    def fake_latent():
        return np.abs(choice(samples[0])).T

    def fake_audio():
        sample = choice(samples[0])
        return from_latent_space(sample)

    trainer.register_batch_complete_callback(batch_complete)

    app = zounds.GanTrainingMonitorApp(
        trainer=trainer,
        model=sound_cls,
        visualization_feature=sound_cls.windowed,
        audio_feature=sound_cls.ogg,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    with app.start_in_thread(args.port):
        Gan.process(
            dataset=(sound_cls, sound_feature),
            trainer=trainer,
            nsamples=int(1e5),
            dtype=np.float32)

    app.start(args.port)

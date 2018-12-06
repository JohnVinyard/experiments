from zounds.learn.trainer import Trainer
from zounds.learn import \
    WassersteinCriticLoss, WassersteinGradientPenaltyLoss, \
    LearnedWassersteinLoss
from torch.optim import Adam
from itertools import chain
import torch


class AdversarialAutoencoderTrainer(Trainer):
    def __init__(
            self,
            network,
            distribution,
            autoencoder_loss,
            epochs,
            batch_size,
            n_critic_iterations=5,
            checkpoint_epochs=1):

        super(AdversarialAutoencoderTrainer, self).__init__(
            epochs, batch_size, checkpoint_epochs)

        self.autoencoder_loss = autoencoder_loss
        self.n_critic_iterations = n_critic_iterations
        self.distribtion = distribution
        self.network = network

        # generator loss - ensure that samples generated from the latent space
        # can fool the data critic
        self.adversarial_data_loss = LearnedWassersteinLoss(self.data_critic)

        # encoder loss - ensure that the latent values produced by the encoder
        # can fool the latent critic
        self.adversarial_latent_loss = \
            LearnedWassersteinLoss(self.latent_critic)

        # data critic loss - ensure that the data critic can tell real and
        # fake samples apart
        self.data_critic_loss = WassersteinCriticLoss(self.data_critic)
        self.data_critic_gp = WassersteinGradientPenaltyLoss(self.data_critic)

        # latent critic loss - ensure that the latent critic can tell samples
        # produced by the encoder apart from the prior specified by distribution
        self.latent_critic_loss = WassersteinCriticLoss(self.latent_critic)
        self.latent_critic_gp = \
            WassersteinGradientPenaltyLoss(self.latent_critic)

        # optimizers
        self.autoencoder_optim = Adam(
            chain(self.encoder.parameters(), self.generator.parameters()),
            lr=0.0001,
            betas=(0, 0.9))
        self.encoder_optim = Adam(
            self.encoder.parameters(), lr=0.0001, betas=(0, 0.9))
        self.generator_optim = Adam(
            self.generator.parameters(), lr=0.0001, betas=(0, 0.9))
        self.data_critic_optim = Adam(
            self.data_critic.parameters(), lr=0.0001, betas=(0, 0.9))
        self.latent_critic_optim = Adam(
            self.latent_critic.parameters(), lr=0.0001, betas=(0, 0.9))

        self.original_samples = None
        self.decoded_samples = None
        self.generated_samples = None
        self.latent_samples = None


    def _cuda(self, device=None):
        for k, v in self.__dict__.iteritems():
            try:
                setattr(self, k, v.cuda(device=device))
            except AttributeError:
                pass

    @property
    def generator(self):
        return self.network.generator

    @property
    def encoder(self):
        return self.network.encoder

    @property
    def latent_critic(self):
        return self.network.latent_critic

    @property
    def data_critic(self):
        return self.network.data_critic

    @property
    def all_networks(self):
        return (
            self.generator,
            self.encoder,
            self.latent_critic,
            self.data_critic
        )

    def _latent(self):
        return self._variable(self.distribtion.sample(self.batch_size))

    def _train_autoencoder(self, data):
        self._zero_grad()
        self._optimize(self.encoder, self.generator)
        samples = self._minibatch(data)

        encoded = self.encoder(samples)
        decoded = self.generator(encoded)
        error = \
            self.autoencoder_loss(decoded, samples) \
            # + torch.abs(encoded).sum()
            # + self.adversarial_latent_loss(encoded)

        error.backward()
        self.autoencoder_optim.step()
        self.original_samples = samples.data.cpu().numpy().squeeze()
        # self.decoded_samples = decoded.data.cpu().numpy().squeeze()
        return float(error.data.cpu().numpy())

    def _train_generator(self, data):
        self._zero_grad()
        self._optimize(self.generator)
        latent = self._latent()
        samples = self.generator(latent)
        error = self.adversarial_data_loss(samples)
        error.backward()
        self.generator_optim.step()
        self.generated_samples = samples.data.cpu().numpy().squeeze()
        return float(error.data.cpu().numpy())

    def _train_predictor(self, data):
        self._zero_grad()
        self._optimize(self.encoder)
        latent = self._latent()
        samples = self.generator(latent)
        encoded = self.encoder(samples).squeeze()
        error = ((latent - encoded) ** 2).mean()
        error.backward()
        self.encoder_optim.step()
        self.latent_samples = encoded.data.cpu().numpy().squeeze()
        return float(error.data.cpu().numpy())

    def _train_data_critic(self, data):
        self._zero_grad()
        self._optimize(self.data_critic)
        for i in xrange(self.n_critic_iterations):
            real = self._minibatch(data)
            latent = self._latent()
            fake = self.generator(latent)
            wloss = self.data_critic_loss(real, fake)
            gradient_penalty = self.data_critic_gp(real.data, fake.data)
            error = wloss + gradient_penalty
            error.backward()
            self.data_critic_optim.step()
        return float(error.data.cpu().numpy())

    def _train_latent_critic(self, data):
        self._zero_grad()
        self._optimize(self.latent_critic)
        for i in xrange(self.n_critic_iterations):
            samples = self._minibatch(data)
            encoded = self.encoder(samples)
            self.latent_samples = encoded.data.cpu().numpy().squeeze()
            latent = self._latent()
            wloss = self.latent_critic_loss(latent, encoded)
            gradient_penalty = self.latent_critic_gp(latent.data, encoded.data)
            error = wloss + gradient_penalty
            error.backward()
            self.latent_critic_optim.step()
        return float(error.data.cpu().numpy())

    def _freeze(self, network, require_grad):
        """
        Freeze or unfreeze a particular network
        """
        for p in network.parameters():
            p.requires_grad = require_grad

    def _optimize(self, *to_optimize):
        """
        Specify which network(s) are being optimized, and will require
        gradients.  All other networks are "frozen", i.e., set to not require
        gradients.
        """
        for network in self.all_networks:
            self._freeze(network, network in to_optimize)

    def _training_step(self, epoch, batch, data):
        error_data = dict()

        try:
            # encoding
            # error_data.update(
            #     latent_critic_loss=self._train_latent_critic(data))

            # autoencoder
            error_data.update(autoencoder_loss=self._train_autoencoder(data))

            # GAN
            # error_data.update(data_critic_loss=self._train_data_critic(data))
            # error_data.update(generator_loss=self._train_generator(data))
            # error_data.update(predictor_loss=self._train_predictor(data))

        except RuntimeError as e:
            if 'Assertion' not in e.message:
                raise
            print e.message

        return error_data

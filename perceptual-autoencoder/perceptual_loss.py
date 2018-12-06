import featureflow as ff
import zounds
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from zounds.learn import \
    from_var, DctTransform, model_hash, RawSampleEmbedding, sample_norm
from torch.optim import Adam
import torch
import argparse
from scipy.signal import gaussian
from itertools import chain

samplerate = zounds.SR11025()
window_size = 2048
latent_dim = 128

BaseModel = zounds.windowed(
    wscheme=samplerate * (16, window_size),
    resample_to=samplerate)



@zounds.simple_lmdb_settings('ae', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    pass


class LayerWithAttention(nn.Module):
    def __init__(
            self,
            layer_type,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(LayerWithAttention, self).__init__()
        self.conv = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.gate = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.attention_func = attention_func

    def forward(self, x):
        c = self.conv(x)
        c = sample_norm(c)
        g = self.gate(x)
        g = sample_norm(g)
        out = F.tanh(c) * self.attention_func(g)
        return out


class ConvAttentionLayer(LayerWithAttention):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(ConvAttentionLayer, self).__init__(
            nn.Conv1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func)


class ConvTransposeAttentionLayer(LayerWithAttention):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid):
        super(ConvTransposeAttentionLayer, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func)


class GeneratorWithAttention(nn.Module):
    def __init__(self):
        super(GeneratorWithAttention, self).__init__()
        self.attn_func = F.sigmoid
        self.layers = [
            ConvTransposeAttentionLayer(
                latent_dim, 512, 4, 2, 0, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 512, 8, 4, 2, attention_func=self.attn_func),
            ConvTransposeAttentionLayer(
                512, 256, 8, 4, 2, attention_func=self.attn_func),
        ]
        self.main = nn.Sequential(*self.layers)
        self.final = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)
        self.gate = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        x = self.main(x)
        c = self.final(x)
        g = self.gate(x)
        x = F.sigmoid(g) * c
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n = 256
        self.n = n
        self.main = nn.Sequential(
            nn.Conv1d(1, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, n, 25, 4, padding=12),
            nn.Conv1d(n, 32, 25, 4, padding=12),
        )

        self.linear = nn.Linear(n, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, window_size)
        for m in self.main:
            x = m(x)
            x = F.leaky_relu(x, 0.2)
        x = x.view(-1, 32 * 8)
        x = self.linear(x)
        return x


#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.linear = nn.Linear(latent_dim, 256)
#         n = 256
#         self.main = nn.Sequential(
#             nn.Conv1d(32, n, 25, 1, padding=12),
#             nn.Conv1d(n, n, 25, 1, padding=12),
#             nn.Conv1d(n, n, 25, 1, padding=12))
#
#         self.final = nn.Conv1d(n, 1, 25, 1, padding=12)
#
#     def forward(self, x):
#         x = x.view(-1, latent_dim)
#         x = self.linear(x)
#         x = x.view(-1, 32, 8)
#         x = F.relu(x)
#
#         for m in self.main:
#             x = F.upsample(x, scale_factor=4)
#             x = m(x)
#             x = F.relu(x)
#
#         x = F.upsample(x, scale_factor=4)
#         x = self.final(x)
#         return x


# class LearnedLoss(nn.Module):
#     def __init__(self):
#         super(LearnedLoss, self).__init__()
#         self.encoder = Encoder()
#         self.l1 = nn.Linear(latent_dim * 2, latent_dim)
#         self.l2 = nn.Linear(latent_dim, 1)
#
#     def forward(self, input, target):
#         inp = self.encoder(input)
#         t = self.encoder(target)
#         x = torch.cat([inp, t], dim=-1)
#         x = self.l1(x)
#         x = F.leaky_relu(x, 0.2)
#         x = self.l2(x)
#         return x




class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = GeneratorWithAttention()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def scalar(x):
    return x.data.cpu().numpy().squeeze()


class AdversarialLoss(object):
    def __init__(self, encoder):
        super(AdversarialLoss, self).__init__()
        self.encoder = encoder
        self.optimizer = Adam(self.encoder.parameters(), lr=0.00005)
        self.real_features = None
        self.fake_features = None
        self.input = None
        self.target = None

    def cuda(self):
        self.encoder = self.encoder.cuda()
        return self

    def encode(self, x):
        # TODO: try comparing activations at *every* layer
        x = self.encoder(x)
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def distance(self, x, y):
        return (x - y) ** 2

    def forward(self, input, target):
        # optimize loss function by trying to push the embeddings as far
        # apart as possible
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.zero_grad()

        inp = self.encode(Variable(input.data))
        t = self.encode(Variable(target.data))

        # perform a crude measure of the "information" content of this
        # feature across the batch, to discourage a trivial solution where some
        # features are always on for real examples, and always off for their
        # reconstructions
        feature_variance = (1 / (t.std(dim=0) + 1e-8)).mean()
        error = self.distance(inp, t).mean()
        adv_loss = -error + feature_variance
        adv_loss.backward()
        self.optimizer.step()

        # compute feature-level reconstruction error, trying to pull the
        # embeddings as close together as possible
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.input = input.data.cpu().numpy().squeeze()
        self.target = target.data.cpu().numpy().squeeze()
        self.encoder.zero_grad()
        inp = self.encode(input)
        t = self.encode(target)
        self.fake_features = inp.data.cpu().numpy().squeeze()
        self.real_features = t.data.cpu().numpy().squeeze()
        error = self.distance(inp, t).mean()
        return error

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PerceptualLoss(nn.MSELoss):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        scale = zounds.BarkScale(
            zounds.FrequencyBand(50, samplerate.nyquist - 300),
            n_bands=512)
        scale.ensure_overlap_ratio(0.5)

        self.scale = scale
        basis_size = 512
        self.lap = 2
        self.basis_size = basis_size

        basis = zounds.fir_filter_bank(
            scale, basis_size, samplerate, gaussian(100, 3))

        weights = Variable(torch.from_numpy(basis).float())
        # out channels x in channels x kernel width
        weights = weights.view(len(scale), 1, basis_size).contiguous()
        self.weights = weights.cuda()

        # kernel = np.array([
        #     [0, -1, 0],
        #     [-1, 0, 1],
        #     [0, 1, 0],
        # ])
        # self.kernel = Variable(
        #     torch.from_numpy(kernel).float()).view(1, 1, 3, 3).cuda()

    def _transform(self, x):
        features = F.conv1d(
            x, self.weights, stride=self.lap, padding=self.basis_size)

        # half-wave rectification
        features = F.relu(features)

        # log magnitude
        features = torch.log(1 + features * 10000)

        # gradient/sharpening
        # features = features.view(x.shape[0], 1, self.basis_size, -1)
        # features = F.conv2d(features, self.kernel, stride=1, padding=(1, 1))
        # features = features.view(x.shape[0], self.basis_size, -1)
        #
        # # half-wave rectification again
        # features = F.relu(features)
        #
        # # normalize to [0, 1]
        # example_wise_max = features.view(x.shape[0], -1).max(dim=-1)[0]
        # example_wise_max = example_wise_max.view(x.shape[0], 1, 1)
        # example_wise_max = example_wise_max + 1e-12
        # features = features / example_wise_max

        return features

    def forward(self, input, target):
        input = input.view(-1, 1, window_size)
        target = target.view(-1, 1, window_size)

        input_features = self._transform(input)
        target_features = self._transform(target)

        return super(PerceptualLoss, self).forward(
            input_features, target_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.ObjectStorageSettings(),
        zounds.AppSettings()
    ])
    args = parser.parse_args()

    zounds.ingest(
        zounds.InternetArchive('LucaBrasi2'),
        Sound,
        multi_threaded=True)


    @zounds.object_store_pipeline_settings(
        'AutoEncoderPipeline',
        args.object_storage_region,
        args.object_storage_username,
        args.object_storage_api_key)
    @zounds.infinite_streaming_learning_pipeline
    class AutoEncoderPipeline(ff.BaseModel):
        scaled = ff.Feature(
            zounds.InstanceScaling)

        autoencoder = ff.Feature(
            zounds.PyTorchAutoEncoder,
            trainer=ff.Var('trainer'),
            needs=scaled)


    try:
        network = AutoEncoderPipeline.load_network()
        print 'loaded network with hash', model_hash(network)
    except RuntimeError as e:
        print e
        network = AutoEncoder()
        for parameter in network.parameters():
            parameter.data.normal_(0, 0.02)

    loss = nn.MSELoss()

    trainer = zounds.SupervisedTrainer(
        model=network,
        loss=loss,
        optimizer=lambda model: Adam(
            chain(model.encoder.parameters(), model.decoder.parameters()),
            lr=0.00005),
        checkpoint_epochs=1,
        epochs=100,
        batch_size=32,
        holdout_percent=0.25)

    app = zounds.SupervisedTrainingMonitorApp(
        trainer=trainer,
        model=Sound,
        visualization_feature=Sound.windowed,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

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


        AutoEncoderPipeline.process(
            dataset=(Sound, Sound.windowed),
            nsamples=int(1e5),
            dtype=np.float32,
            trainer=trainer)

    app.start(8888)

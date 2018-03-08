import argparse
from itertools import chain

import featureflow as ff
import numpy as np
import zounds
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from zounds.learn import from_var, model_hash, sample_norm, apply_network
from zounds.learn import GatedConvLayer, GatedConvTransposeLayer
import torch
from torch.autograd import Variable
from scipy.signal import gaussian

samplerate = zounds.SR11025()
window_size = 2048
latent_dim = 128

BaseModel = zounds.windowed(
    wscheme=samplerate * (16, window_size),
    resample_to=samplerate)


@zounds.simple_lmdb_settings('ae', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    pass


class GeneratorWithAttention(nn.Module):
    def __init__(self):
        super(GeneratorWithAttention, self).__init__()
        self.attn_func = F.sigmoid
        self.layers = [
            self._make_layer(latent_dim, 512, 4, 2, 0),
            self._make_layer(512, 512, 8, 4, 2),
            self._make_layer(512, 512, 8, 4, 2),
            self._make_layer(512, 512, 8, 4, 2),
        ]
        self.main = nn.Sequential(*self.layers)
        self.final = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)
        self.gate = nn.ConvTranspose1d(256, 1, 16, 8, 4, bias=False)

    def _make_layer(self, in_channels, out_channels, kernel, stride, padding):
        return GatedConvTransposeLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
            attention_func=self.attn_func,
            norm=sample_norm)

    def forward(self, x):
        x = x.view(-1, latent_dim, 1)
        x = self.main(x)
        c = self.final(x)
        g = self.gate(x)
        x = self.attn_func(g) * c
        return x


class AnalyzerWithAttention(nn.Module):
    def __init__(self):
        super(AnalyzerWithAttention, self).__init__()
        self.attn_func = F.sigmoid
        self.first = nn.Conv1d(1, 256, 16, 8, 4, bias=False)
        self.gate = nn.Conv1d(1, 256, 16, 8, 4, bias=False)
        self.layers = [
            self._make_layer(256, 512, 8, 4, 2),
            self._make_layer(512, 512, 8, 4, 2),
            self._make_layer(512, 512, 8, 4, 2),
            self._make_layer(512, 512, 4, 2, 0)
        ]
        self.main = nn.Sequential(*self.layers)
        self.linear = nn.Linear(512, latent_dim, bias=False)

    def _make_layer(self, in_channels, out_channels, kernel, stride, padding):
        return GatedConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
            attention_func=self.attn_func,
            norm=sample_norm)

    def forward(self, x):
        x = x.view(-1, 1, window_size)
        c = self.first(x)
        g = self.gate(x)
        x = self.attn_func(g) * c
        x = self.main(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


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

    def _transform(self, x):
        features = F.conv1d(
            x, self.weights, stride=self.lap, padding=self.basis_size)

        # half-wave rectification
        features = F.relu(features)

        # log magnitude
        features = torch.log(1 + features * 10000)

        return features

    def forward(self, input, target):
        input = input.view(-1, 1, window_size)
        target = target.view(-1, 1, window_size)

        input_features = self._transform(input)
        target_features = self._transform(target)

        return super(PerceptualLoss, self).forward(
            input_features, target_features)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = AnalyzerWithAttention()
        self.decoder = GeneratorWithAttention()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def reconstruct():
    snd = Sound.random()

    # sliding window
    _, windowed = snd.resampled.sliding_window_with_leftovers(
        window_size, window_size // 2, dopad=True)

    # instance scaling
    max = (windowed.max(axis=-1, keepdims=True) + 1e-8)
    windowed /= max

    # encode
    encoded = apply_network(network.encoder, windowed, chunksize=64)

    # decode
    decoded = apply_network(network.decoder, encoded, chunksize=64)
    decoded = decoded.squeeze()
    decoded *= np.hanning(window_size) * max
    decoded = zounds.ArrayWithUnits.from_example(decoded, windowed)

    synth = zounds.WindowedAudioSynthesizer()
    recon = synth.synthesize(decoded)
    return recon


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
        help='which loss to use: (perceptual|mse)',
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
        network = AutoEncoder()
        for parameter in network.parameters():
            parameter.data.normal_(0, 0.02)

    loss = nn.MSELoss() if args.loss == 'mse' else PerceptualLoss()

    trainer = zounds.SupervisedTrainer(
        model=network,
        loss=loss,
        optimizer=lambda model: Adam(
            chain(model.encoder.parameters(), model.decoder.parameters()),
            lr=0.00005),
        checkpoint_epochs=1,
        epochs=50,
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

    if args.reconstruct:
        recon = reconstruct()
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
                trainer=trainer)

    app.start(8888)

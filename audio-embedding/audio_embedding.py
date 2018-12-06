from __future__ import division
import zounds
import numpy as np
import featureflow as ff
from torch import nn
from torch.nn import functional as F
from torch import optim
from scipy.spatial.distance import cdist
from zounds.learn.util import to_var, from_var
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)


def one_hot(x):
    # mu law encode
    x = zounds.mu_law(x, mu=255)
    # quantize
    x = (255 * ((x * 0.5) + 0.5))
    x = x.astype(np.int64)
    x = zounds.ArrayWithUnits(x, x.dimensions)
    return x


def inverse_one_hot(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2
    x = zounds.inverse_mu_law(x)
    x = zounds.AudioSamples(x, samplerate)
    return x


# def inverse_one_hot(x):
#     x = x.reshape((-1, 256, 8192))
#     indices = np.argmax(x, axis=1).astype(np.float32)
#     indices /= 255.
#     indices = (indices - 0.5) * 2
#     return indices


@zounds.simple_lmdb_settings(
    'audio_embedding', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    oh = zounds.ArrayWithUnitsFeature(
        one_hot,
        needs=BaseModel.resampled)

    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=samplerate.frequency,
            duration=samplerate.frequency * 2),
        wfunc=None,
        needs=oh)


class AudioEmbedding(nn.Module):
    def __init__(self):
        super(AudioEmbedding, self).__init__()
        self.dimensions = 3
        self.embedding = nn.Embedding(256, self.dimensions, max_norm=1.0)
        self.linear = nn.Linear(self.dimensions, 256, bias=False)

    def embed(self, x):
        x = x.long()
        return self.embedding(x)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = self.linear(x)
        x = F.log_softmax(x)
        return x


BasePipeline = zounds.learning_pipeline()


@zounds.simple_settings
class AudioEmbeddingPipeline(BasePipeline):
    network = ff.PickleFeature(
        zounds.PyTorchNetwork,
        trainer=zounds.SupervisedTrainer(
            model=AudioEmbedding(),
            loss=nn.NLLLoss(),
            optimizer=lambda model:
            optim.Adam(model.parameters(), lr=0.00005),
            epochs=100,
            batch_size=256,
            holdout_percent=0.2),
        needs=BasePipeline.samples,
        # TODO: Resolve inconsistent
        # MRO when using KeySelector and AspectExtractor
        training_set_prep=lambda data: data['samples'])

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(network,),
        store=True)



if __name__ == '__main__':
    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)


    def gen():
        for snd in Sound:
            yield dict(data=snd.windowed[..., 0], labels=snd.windowed[..., 1])


    if not AudioEmbeddingPipeline.exists():
        AudioEmbeddingPipeline.process(
            samples=gen(),
            nsamples=int(1e7),
            dtype=np.int64)

    aep = AudioEmbeddingPipeline()
    network = aep.pipeline[0].network


    def embed_samples(x):
        """
        transform raw audio samples to embeddings
        """
        x = one_hot(x)
        x = to_var(x)
        embedded = network.embedding(x)
        embedded = from_var(embedded)
        return embedded


    def unembed_samples(x):
        """
        transform embedded samples to raw audio samples
        """
        w = network.embedding.weight.cpu().data.numpy()
        dist = cdist(w, x)
        indices = np.argmin(dist, axis=0)
        return inverse_one_hot(indices)


    def plot_embedding():
        q = torch.arange(0, 255).long()
        q = Variable(q).cuda()
        embedded = network.embedding(q)
        points = from_var(embedded)
        print points.shape

        fig = plt.figure()
        dim = points.shape[-1]
        ax = fig.add_subplot(111, projection='3d')
        coords = [points[:, i] for i in xrange(dim)]
        x = ax.scatter(*coords)
        fig = x.get_figure()
        fig.set_size_inches((50, 50))

        # for i, p in enumerate(points):
        #     x.axes.annotate(str(i), tuple(points[i]), fontsize=50)


        plt.savefig('embedding')


    # TODO: Look at the norms of all embeddings.  Should training include a
    # forced unit norm before the linear dimension?
    # TODO: function to go from embedding to raw samples
    # TODO: listen to mu-law encoded and quantized samples
    # TODO: visualize the 3d embedding
    # TODO: add noise in the raw audio sample domain and listen
    # TODO: add noise in the embedding domain and listen

    plot_embedding()
    snd = Sound.random()

    weights = network.embedding.weight.cpu().data.numpy()
    norms = np.linalg.norm(weights, axis=-1)

    samples = zounds.AudioSamples(snd.resampled[:int(samplerate)], samplerate)
    oh = one_hot(samples)
    inverted = inverse_one_hot(oh)

    embedded = embed_samples(samples)
    recon = unembed_samples(embedded)

    samples_with_noise = samples + np.random.normal(0, 0.01, samples.shape)
    embedding_with_noise = embedded + np.random.normal(0, 0.02, embedded.shape)
    recon_with_noise = unembed_samples(embedding_with_noise)

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())
    app.start(8888)

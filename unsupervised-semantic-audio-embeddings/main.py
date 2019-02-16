from __future__ import division, print_function
import zounds
import argparse
from data import dataset
from deformations import make_pitch_shift, make_time_stretch, additive_noise
from training_data import TripletSampler
from train import Trainer
from network import EmbeddingNetwork
import torch
import numpy as np
import cPickle as pickle
from search import TreeSearch

# resample all audio in our dataset to this rate
samplerate = zounds.SR11025()

# produce a base class for our audio processing graph, which will do some
# basic preprocessing and transcoding of the signal
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

# the length in samples of the audio segments we'll be creating embeddings for
window_size_samples = 8192
slice_duration = samplerate.frequency * window_size_samples

# segments occurring within ten seconds of our anchor will be considered
# semantically similar
temporal_proximity = zounds.Seconds(10)

# a collection of the audio deformations we'll use during training.  Temporal
# proximity is included implicitly
deformations = [
    make_time_stretch(samplerate, window_size_samples),
    make_pitch_shift(samplerate),
    additive_noise
]


@zounds.simple_lmdb_settings(
    '/hdd/sounddb2', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    """
    An audio processing graph, that will resample each audio file to 11025hz
    and store the results in an LMDB database
    """
    short_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.HalfLapped(),
        wfunc=zounds.OggVorbisWindowingFunc(),
        needs=BaseModel.resampled)

    stft = zounds.ArrayWithUnitsFeature(
        zounds.FFT,
        needs=short_windowed)


def train(network, batch_size, device, checkpoint, weights_file_path):
    """
    Train the model indefinitely
    """
    sampler = TripletSampler(
        Sound, slice_duration, deformations, temporal_proximity)
    trainer = Trainer(
        network=network,
        triplet_sampler=sampler,
        learning_rate=1e-4,
        batch_size=batch_size).to(device)

    for batch_num, error in enumerate(trainer.train()):
        print('Batch: {batch_num}, Error: {error}'.format(**locals()))
        if batch_num % checkpoint == 0:
            torch.save(network.state_dict(), weights_file_path)


def compute_all_embeddings(network):
    """
    A generator that will compute embeddings for every non-overlapping segment
    of duration window_size_samples in the database
    """
    for snd in Sound:
        windowed = snd.resampled.sliding_window(
            samplerate * window_size_samples).astype(np.float32)
        arr = zounds.learn.apply_network(
            network, windowed, chunksize=64)
        ts = zounds.ArrayWithUnits(
            arr, [windowed.dimensions[0], zounds.IdentityDimension()])
        print(snd._id)
        yield snd._id, ts


def build_search_index(network, search_file_path):
    """
    Build both a brute force search index, as well as an index that uses a tree
    of random hyperplane splits
    """
    try:
        with open(search_file_path, 'rb') as f:
            search = pickle.load(f)
    except IOError:
        search = zounds.BruteForceSearch(
            compute_all_embeddings(network), distance_metric='cosine')
        with open(search_file_path, 'wb') as f:
            pickle.dump(search, f, pickle.HIGHEST_PROTOCOL)

    print('building tree...')
    tree_search = TreeSearch(search)
    return search, tree_search


def demo_negative_mining(network, batch_size, device):
    from matplotlib import pyplot as plt, gridspec
    from itertools import product

    sampler = TripletSampler(
        Sound, slice_duration, deformations, temporal_proximity)
    trainer = Trainer(
        network=network,
        triplet_sampler=sampler,
        learning_rate=1e-4,
        batch_size=batch_size).to(device)

    spec = gridspec.GridSpec(4, 4, wspace=0.25, hspace=0.25)
    fig = plt.figure(figsize=(15, 15))

    for x, y in product(xrange(4), xrange(4)):
        anchor_to_positive, anchor_to_negative, mined_anchor_to_negative = \
            trainer.negative_mining_demo()

        ax = plt.subplot(spec[x, y])
        ax.plot(anchor_to_positive, label='anchor-to-positive')
        ax.plot(anchor_to_negative, label='anchor-to-negative')
        ax.plot(mined_anchor_to_negative, label='mined-anchor-to-negative')
        ax.set_xticks([])
        ax.set_ylim(0, 1.0)

    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.savefig(
        'negative_mining.png'.format(**locals()), format='png')
    fig.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        zounds.ui.AppSettings()
    ])
    parser.add_argument(
        '--ingest',
        help='should data be ingested',
        action='store_true')
    parser.add_argument(
        '--batch-size',
        help='Batch size to use when training',
        type=int)
    parser.add_argument(
        '--checkpoint',
        help='save network weights every N batches',
        type=int)
    parser.add_argument(
        '--weights-file-path',
        help='the name of the file where weights should be saved')
    parser.add_argument(
        '--search',
        help='test the search',
        action='store_true')
    parser.add_argument(
        '--search-file-path',
        help='the path where a pre-built search should be stored',
        required=False)
    parser.add_argument(
        '--demo-negative-mining',
        help='run a demo of within-batch semi-hard negative mining',
        action='store_true')

    args = parser.parse_args()

    if args.ingest:
        zounds.ingest(dataset, Sound, multi_threaded=True)

    network, device = EmbeddingNetwork.load_network(args.weights_file_path)

    if args.search:
        search, tree_search = build_search_index(
            network=network,
            search_file_path=args.search_file_path)
    elif args.demo_negative_mining:
        demo_negative_mining(network, args.batch_size, device)
    else:
        train(
            network=network,
            batch_size=args.batch_size,
            device=device,
            checkpoint=args.checkpoint,
            weights_file_path=args.weights_file_path)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.stft,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())
    app.start(port=args.port)

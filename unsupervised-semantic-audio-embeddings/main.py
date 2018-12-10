from __future__ import division, print_function
import zounds
import argparse
from data import dataset
from deformations import make_pitch_shift, make_time_stretch, additive_noise
from training_data import TripletSampler
from train import Trainer
from network import EmbeddingNetwork
import torch

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


@zounds.simple_lmdb_settings('sounds', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    """
    An audio processing graph, that will resample each audio file to 11025hz
    and store the results in an LMDB database
    """
    pass


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
    args = parser.parse_args()

    if args.ingest:
        zounds.ingest(dataset, Sound, multi_threaded=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = EmbeddingNetwork().to(device)

    try:
        # load network weights from a file on disk
        state_dict = torch.load(args.weights_file_path)
        network.load_state_dict(state_dict)
    except IOError:
        # There were no weights stored on disk.  Initialize them
        network.initialize_weights()

    sampler = TripletSampler(
        Sound, slice_duration, deformations, temporal_proximity)
    trainer = Trainer(
        network=network,
        triplet_sampler=sampler,
        learning_rate=1e-4,
        batch_size=args.batch_size).to(device)

    for batch_num, error in enumerate(trainer.train()):
        print('Batch: {batch_num}, Error: {error}'.format(**locals()))
        if batch_num % args.checkpoint == 0:
            torch.save(network.state_dict(), args.weights_file_path)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.resampled,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())
    app.start(port=args.port)

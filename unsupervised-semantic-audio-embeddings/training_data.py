from __future__ import print_function
import zounds
import numpy as np


class TripletSampler(object):
    """
    Sample (anchor, positive, negative) triplets from a zounds database
    """

    def __init__(
            self,
            sound_cls,
            slice_duration,
            deformations,
            temporal_proximity):

        super(TripletSampler, self).__init__()
        self.temporal_proximity = temporal_proximity
        self._temporal_proximity_sentinel = object()
        self.deformations = \
            list(deformations) + [self._temporal_proximity_sentinel]
        self.slice_duration = slice_duration
        self.sound_cls = sound_cls

        # compute the weighted probabilities of choosing segments from each
        # sound based on its length.  Longer sounds will be chosen with a higher
        # likelihood than shorter sounds, such that every segment in the
        # database has an equal probability of being chosen
        items = [(snd._id, snd.resampled.end) for snd in self.sound_cls]
        durations = np.array([item[1] / zounds.Seconds(1) for item in items])
        probabilities = durations / durations.sum()

        self.probabilities = probabilities
        self.items = [item[0] for item in items]
        self.durations = dict(items)

    def _get_samples(self, sound_id, start_ps, pad=False):
        """
        Fetch the segment from sound_id starting at start_ps (picoseconds),
        zero-padded when appropriate
        """
        snd = self.sound_cls(_id=sound_id)
        start = zounds.Picoseconds(int(start_ps))
        time_slice = zounds.TimeSlice(start=start, duration=self.slice_duration)
        max_samples = \
            int(self.slice_duration / snd.resampled.samplerate.frequency)
        samples = snd.resampled[time_slice][:max_samples]
        sample_length = len(samples)

        if sample_length < max_samples:
            if pad:
                diff = max_samples - sample_length
                padded = np.pad(
                    samples, (0, diff), mode='constant', constant_values=0)
                samples = zounds.ArrayWithUnits(padded, samples.dimensions)
            else:
                raise ValueError(
                    '{sound_id} has only {sample_length} samples, '
                    'but {max_samples} are required'.format(**locals()))

        return time_slice, samples

    def _random_sound(self):
        """
        Choose a sound at random based on the weighted probabilities computed in
        __init__
        """
        _id = np.random.choice(self.items, p=self.probabilities)
        duration = self.durations[_id]
        duration_ps = duration / zounds.Picoseconds(1)
        return _id, duration, duration_ps

    def _sample_slice(self, pad=False):
        """
        Choose an audio slice, at random, padding when necessary
        """
        slice_duration_ps = self.slice_duration / zounds.Picoseconds(1)

        _id, duration, duration_ps = self._random_sound()

        if not pad:
            # we can't pad, so keep searching for a sound that is at least as
            # long as our desired duration
            while duration_ps < slice_duration_ps:
                _id, duration, duration_ps = self._random_sound()

        start_ps = np.random.uniform(0, duration_ps - slice_duration_ps)
        time_slice, samples = self._get_samples(_id, start_ps, pad=pad)
        return SoundSlice(_id, duration, time_slice, samples)

    def _sample_proximal(self, sound_slice):
        """
        Sample temporally proximal (near-in-time) samples, given an anchor
        sample
        """
        min_start = zounds.Seconds(0)
        start = max(min_start, sound_slice.start - self.temporal_proximity)
        start_ps = start / zounds.Picoseconds(1)

        max_end = sound_slice.sound_duration - self.slice_duration
        end = min(max_end, sound_slice.end + self.temporal_proximity)
        end_ps = end / zounds.Picoseconds(1)

        proximal_start_ps = np.random.uniform(start_ps, end_ps)
        _, samples = self._get_samples(
            sound_slice.sound_id, proximal_start_ps, pad=False)
        return samples

    def sample(self, batch_size):
        """
        Sample batch_size triplets
        """

        # choose a deformation
        deformation = np.random.choice(self.deformations)
        is_temporal_proximity_batch = \
            deformation is self._temporal_proximity_sentinel

        anchors = []
        positives = []
        negatives = []

        for _ in xrange(batch_size):
            # choose positive and negative examples randomly
            anchor_sound_slice = self._sample_slice(
                pad=not is_temporal_proximity_batch)
            anchors.append(anchor_sound_slice.samples)

            negative_sound_slice = self._sample_slice(
                pad=not is_temporal_proximity_batch)
            negatives.append(negative_sound_slice.samples)

            if is_temporal_proximity_batch:
                # if we've chosen the "temporal proximity" deformation, find
                # a nearby segment for our positive example
                positive_samples = self._sample_proximal(anchor_sound_slice)
                positives.append(positive_samples)

        time_dimension = anchors[0].dimensions[0]
        identity_dim = zounds.IdentityDimension()

        anchors = zounds.ArrayWithUnits(
            np.vstack(anchors), [identity_dim, time_dimension])
        negatives = zounds.ArrayWithUnits(
            np.vstack(negatives), [identity_dim, time_dimension])

        if not is_temporal_proximity_batch:
            # deform the anchor to derive the positive example
            positives = deformation(anchors)
        else:
            # positive examples have already been drawn from samples that occur
            # near in time to the anchor examples
            positives = zounds.ArrayWithUnits(
                np.vstack(positives), [identity_dim, time_dimension])

        # package up the batch and ship it
        batch = np.stack([anchors, positives, negatives], axis=1)
        batch = zounds.ArrayWithUnits(
            batch, [identity_dim, identity_dim, time_dimension])
        return batch


class SoundSlice(object):
    """
    Convenience class that encapsulates the notion of a segment or slice of
    audio samples, exposing the samples themselves, as well as their "address"
    in their originating audio file
    """
    def __init__(self, sound_id, sound_duration, time_slice, samples):
        super(SoundSlice, self).__init__()
        self.samples = samples
        self.time_slice = time_slice
        self.sound_duration = sound_duration
        self.sound_id = sound_id

    @property
    def start(self):
        return self.time_slice.start

    @property
    def duration(self):
        return self.time_slice.duration

    @property
    def end(self):
        return self.time_slice.end

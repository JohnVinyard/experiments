from __future__ import print_function
import zounds
import numpy as np


class SoundSlice(object):
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


class TripletSampler(object):
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
        items = [(snd._id, snd.resampled.end) for snd in self.sound_cls]
        durations = np.array([item[1] / zounds.Seconds(1) for item in items])
        probabilities = durations / durations.sum()
        self.probabilities = probabilities
        self.items = [item[0] for item in items]
        self.durations = dict(items)

    def _get_samples(self, sound_id, start_ps):
        snd = self.sound_cls(_id=sound_id)
        start = zounds.Picoseconds(int(start_ps))
        time_slice = zounds.TimeSlice(start=start, duration=self.slice_duration)
        max_samples = \
            int(self.slice_duration / snd.resampled.samplerate.frequency)
        return time_slice, snd.resampled[time_slice][:max_samples]

    def _sample_slice(self):
        _id = np.random.choice(self.items, p=self.probabilities)
        duration = self.durations[_id]
        duration_ps = duration / zounds.Picoseconds(1)
        slice_duration_ps = self.slice_duration / zounds.Picoseconds(1)
        start_ps = np.random.uniform(0, duration_ps - slice_duration_ps)
        time_slice, samples = self._get_samples(_id, start_ps)
        return SoundSlice(_id, duration, time_slice, samples)

    def _sample_proximal(self, sound_slice):
        min_start = zounds.Seconds(0)
        start = max(min_start, sound_slice.start - self.temporal_proximity)
        start_ps = start / zounds.Picoseconds(1)

        max_end = sound_slice.sound_duration - self.slice_duration
        end = min(max_end, sound_slice.end + self.temporal_proximity)
        end_ps = end / zounds.Picoseconds(1)

        proximal_start_ps = np.random.uniform(start_ps, end_ps)
        _, samples = self._get_samples(sound_slice.sound_id, proximal_start_ps)
        return samples

    def sample(self, batch_size):
        deformation = np.random.choice(self.deformations)
        is_temporal_proximity_batch = \
            deformation is self._temporal_proximity_sentinel

        anchors = []
        positives = []
        negatives = []

        for _ in xrange(batch_size):

            anchor_sound_slice = self._sample_slice()
            anchors.append(anchor_sound_slice.samples)

            negative_sound_slice = self._sample_slice()
            negatives.append(negative_sound_slice.samples)

            if is_temporal_proximity_batch:
                positive_samples = self._sample_proximal(anchor_sound_slice)
                positives.append(positive_samples)

        time_dimension = anchors[0].dimensions[0]
        identity_dim = zounds.IdentityDimension()

        anchors = zounds.ArrayWithUnits(
            np.vstack(anchors), [identity_dim, time_dimension])
        negatives = zounds.ArrayWithUnits(
            np.vstack(negatives), [identity_dim, time_dimension])

        if not is_temporal_proximity_batch:
            positives = deformation(anchors)
        else:
            positives = zounds.ArrayWithUnits(
                np.vstack(positives), [identity_dim, time_dimension])

        batch = np.stack([anchors, positives, negatives], axis=1)
        batch = zounds.ArrayWithUnits(
            batch, [identity_dim, identity_dim, time_dimension])
        return batch


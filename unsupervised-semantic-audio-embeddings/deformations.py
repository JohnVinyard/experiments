import numpy as np
import zounds


def additive_noise(anchor):
    """
    Add broadband gaussian noise to a signal
    """
    amt = np.random.uniform(0.01, 0.05)
    return anchor + np.random.normal(0, amt, anchor.shape).astype(anchor.dtype)


def make_time_stretch(samplerate, window_size_samples):
    """
    Produce a time stretch function, given the samplerate and
    size in samples of the incoming signals
    """

    def time_stretch(anchor):
        """
        Change the duration of a batch of sounds without changing
        their pitch
        """
        factor = np.random.uniform(0.5, 2.0)
        anchor = zounds.ArrayWithUnits(
            anchor.squeeze(),
            [zounds.IdentityDimension(), zounds.TimeDimension(*samplerate)])

        # change the duration of the audio
        stretched = zounds.spectral.time_stretch(anchor, factor)

        # zero-pad or truncate the samples as necessary, to ensure
        # they are still window_size_sampes in the time dimension
        if factor > 1:
            diff = window_size_samples - stretched.shape[-1]
            return np.pad(
                stretched,
                ((0, 0), (0, diff)),
                mode='constant',
                constant_values=0)
        elif factor < 1:
            return stretched[:, :window_size_samples]
        else:
            return stretched
    return time_stretch


def make_pitch_shift(samplerate):
    """
    Produce a pitch shift function, given the samplerate
    of the incoming samples
    """

    def pitch_shift(anchor):
        """
        Change the pitch of a batch of samples, without changing their duration
        """
        anchor = zounds.ArrayWithUnits(
            anchor.squeeze(),
            [zounds.IdentityDimension(), zounds.TimeDimension(*samplerate)])

        # raise or lower the pitch of the batch by up to 10 semitones
        amt = np.random.randint(-10, 10)
        shifted = zounds.spectral.pitch_shift(anchor, amt)
        return shifted

    return pitch_shift



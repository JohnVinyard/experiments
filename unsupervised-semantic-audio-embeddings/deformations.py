import numpy as np
import zounds


def additive_noise(anchor):
    amt = np.random.uniform(0.01, 0.05)
    return anchor + np.random.normal(0, amt, anchor.shape).astype(anchor.dtype)


def make_time_stretch(samplerate, window_size_samples):
    def time_stretch(anchor):
        factor = np.random.uniform(0.5, 2.0)
        anchor = zounds.ArrayWithUnits(
            anchor.squeeze(),
            [zounds.IdentityDimension(), zounds.TimeDimension(*samplerate)])
        stretched = zounds.spectral.time_stretch(anchor, factor)
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
    def pitch_shift(anchor):
        anchor = zounds.ArrayWithUnits(
            anchor.squeeze(),
            [zounds.IdentityDimension(), zounds.TimeDimension(*samplerate)])
        amt = np.random.randint(-10, 10)
        shifted = zounds.spectral.pitch_shift(anchor, amt)
        return shifted
    return pitch_shift
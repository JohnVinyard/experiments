import zounds
import numpy as np
from scipy.signal import morlet
from scipy.fftpack import dct
from numpy.lib.stride_tricks import as_strided
import featureflow as ff

samplerate = zounds.SR11025()

BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

window_sr = zounds.SampleRate(
    frequency=samplerate.frequency * 256,
    duration=samplerate.frequency * 512)

scale = zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist), 512)
small_scale = zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist), 96)
chroma_scale = zounds.ChromaScale(zounds.FrequencyBand(20, samplerate.nyquist))
summary_scale = zounds.MelScale(
    zounds.FrequencyBand(20, samplerate.nyquist), 13)

second_sr = zounds.SampleRate(
    frequency=window_sr.frequency * 42,
    duration=window_sr.frequency * 42)


def filter_bank(kernel_size=512, scale=scale):
    basis = np.zeros((len(scale), kernel_size), dtype=np.complex128)

    for i, band in enumerate(scale):
        center = band.center_frequency
        basis[i] = morlet(kernel_size, 25,
                          (center / samplerate.nyquist) * 2 * np.pi)
    return basis


FILTER_BANK = filter_bank()
SMALLER_FILTER_BANK = filter_bank(scale=small_scale)


def spec(windowed):
    # compute log amplitude mel spectrogram
    x = np.dot(FILTER_BANK, windowed.T)
    x = np.abs(x)
    x = 20 * np.log10(x + 1)
    x = x.T
    x = zounds.ArrayWithUnits(x, [
        zounds.TimeDimension(*window_sr),
        zounds.FrequencyDimension(scale)
    ])
    x = x * zounds.AWeighting()
    return x


def unit_norm(x):
    original_shape = x.shape
    x = x.reshape((x.shape[0], -1))
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    normed = x / (norms + 1e-8)
    return normed.reshape(original_shape)


def compute_chroma(spec):
    x = chroma_scale.apply(spec, zounds.spectral.IdentityWindowingFunc())
    x = unit_norm(x)
    return zounds.ArrayWithUnits(
        x, [spec.dimensions[0], zounds.IdentityDimension()])


def compute_mfcc(spec):
    coeffs = dct(spec, axis=1)[:, 1: 13]
    coeffs = unit_norm(coeffs)
    return zounds.ArrayWithUnits(
        coeffs, [spec.dimensions[0], zounds.IdentityDimension()])


def compute_bands(spec):
    x = summary_scale.apply(spec, zounds.spectral.IdentityWindowingFunc())
    x = unit_norm(x)
    return zounds.ArrayWithUnits(
        x, [spec.dimensions[0], zounds.FrequencyDimension(summary_scale)])


def compute_features(spec):
    # a = compute_chroma(spec)
    b = compute_mfcc(spec)
    c = compute_bands(spec)
    x = np.concatenate([b, c], axis=1)
    return zounds.ArrayWithUnits(
        x, [spec.dimensions[0], zounds.IdentityDimension()])


def sliding_window(x, kernel, step):
    # pad the data
    kernel = np.array(kernel)
    step = np.array(step)
    overlap = kernel - step
    strides = np.array(x.strides)

    shape = np.array(x.shape)
    nsteps = (shape // step) * step
    desired_size = nsteps + overlap

    diff = np.abs(desired_size - shape)

    samples = np.pad(x, [(0, d) for d in diff], mode='constant')

    # window the data
    n = np.clip(1 + (shape - kernel) // step, 1, np.inf).astype(np.int32)
    new_shape = tuple(n) + tuple(kernel)
    new_strides = tuple(strides * step) + tuple(strides)
    windowed = as_strided(samples, new_shape, new_strides)
    return n, windowed


random = np.random.RandomState(seed=2)


def weights(shape):
    x = random.normal(0, 1, (shape[0], np.product(shape[1:])))
    u, _, v = np.linalg.svd(x, full_matrices=False)
    x = u if u.shape == x.shape else v
    return x.reshape(shape)

# out channels x length x in channels
W1 = weights((512, 2, 25))
W2 = weights((512, 2, 512))
W3 = weights((512, 2, 512))
W4 = weights((512, 2, 512))

all_weights = [W1, W2, W3, W4]


def convolve1d(signal, w, step):
    print signal.shape

    # signal is (batch, length, channels)
    kernel = w.shape[1]
    out_channels = w.shape[0]
    n, windowed = sliding_window(signal, (1, kernel, 1), (1, step, 1))
    flattened = windowed.reshape((
        np.product(windowed.shape[:2]),
        np.product(windowed.shape[2:])))
    activations = np.dot(w.reshape((out_channels, -1)), flattened.T).T
    with_time = activations.reshape(windowed.shape[:2] + (out_channels,))
    return with_time


def pooling(signal, kernel, step):
    print signal.shape

    # signal is (batch, length, channels)
    n, windowed = sliding_window(signal, (1, kernel, 1), (1, step, 1))
    pooled = np.amax(windowed, axis=(3, 4, 5))
    return pooled


def relu(signal):
    return np.clip(signal, 0, np.inf)


class SpectrogramFeatures(ff.Node):
    def __init__(self, needs=None):
        super(SpectrogramFeatures, self).__init__(needs=needs)
        self._cache = None

    def _enqueue(self, data, pusher):
        if self._cache is None:
            self._cache = data
        else:
            self._cache = np.concatenate([self._cache, data])

    def _dequeue(self):
        if not self._finalized:
            raise ff.NotEnoughData()

        return self._cache

    def _process(self, data):
        x = np.array(data)

        for weight in all_weights:
            x = convolve1d(x, weight, 1)
            x = relu(x)
            x = pooling(x, 3, 2)

        x = x.reshape((data.shape[0], all_weights[-1].shape[0]))
        x = unit_norm(x)

        yield zounds.ArrayWithUnits(
            x,
            [
                zounds.TimeDimension(*second_sr),
                zounds.IdentityDimension(),
            ])


@zounds.simple_lmdb_settings('random', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=window_sr,
        needs=BaseModel.resampled)

    spectrogram = zounds.ArrayWithUnitsFeature(
        spec,
        needs=windowed)

    spec_features = zounds.ArrayWithUnitsFeature(
        compute_features,
        needs=spectrogram)

    windowed_features = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=second_sr,
        needs=spec_features)

    final_features = zounds.ArrayWithUnitsFeature(
        SpectrogramFeatures,
        needs=windowed_features)


if __name__ == '__main__':
    archive_ids = [
        'AOC11B',
        'CHOPINBallades-NEWTRANSFER',
        '02.LostInTheShadowsLouGramm',
        '08Scandalous',
        'Free_20s_Jazz_Collection',
        'LucaBrasi2',
        'Greatest_Speeches_of_the_20th_Century',
        'rome_sample_pack',
        'CityGenetic',
        'SvenMeyer-KickSamplePack'
    ]
    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.spectrogram,
        audio_feature=Sound.resampled,
        globals=globals(),
        locals=locals())

    print Sound.random().final_features.shape

    with app.start_in_thread(8888):
        for _id in archive_ids:
            zounds.ingest(
                zounds.InternetArchive(_id), Sound, multi_process=True)

    from multiprocessing.pool import ThreadPool as Pool, cpu_count
    pool = Pool(cpu_count())
    gen = pool.imap_unordered(lambda snd: (snd._id, snd.final_features), Sound)
    search = zounds.BruteForceSearch(gen)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.spectrogram,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())
    app.start(8888)







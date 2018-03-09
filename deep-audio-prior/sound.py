import zounds


def sound_cls(samplerate, sample_size):
    window_sample_rate = zounds.SampleRate(
        frequency=samplerate.frequency * sample_size,
        duration=samplerate.frequency * sample_size)

    BaseModel = zounds.windowed(
        window_sample_rate, store_resampled=True, resample_to=samplerate)

    @zounds.simple_lmdb_settings(
        'prior', map_size=1e11, user_supplied_id=True)
    class Sound(BaseModel):
        long_windowed = zounds.ArrayWithUnitsFeature(
            zounds.SlidingWindow,
            wscheme=window_sample_rate,
            needs=BaseModel.resampled)

    return Sound

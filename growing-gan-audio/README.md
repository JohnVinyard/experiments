# Progressively Growing GANs for Audio Generation

In [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196),
the authors improve on state of the art image size and quality by slowing moving
the generator and discriminator through a curriculum of ever-larger image
resolutions.

## Experiment Details

Code for the experiment is [here](growing-gan.py).

Intuitively, this seemed promising for audio (especially musical) signals, given
the importance of their structure at many different time scales.

This experiment adapts the growing GAN training methodology to one-dimensional
audio, and attempts to produce ~740ms 11025hz samples by learning to produce
 realistic samples at ever-larger sampling rates.

The audio corpus for this experiment is
[Johann Sebastian Bach - Complete Partitas for piano - Vol. I](https://archive.org//details/AOC11B).

Results are noisy and unsatisfying, but not completely without structure.  There
are hints of piano flourishes and melodic lines buried in the noise.

Spectrograms of generated audio, and ogg vorbis files with some generated
samples can be found [here](samples).

## Future Investigations

- Try this on a simpler dataset (NSynth?)
- While noisy, the samples _do_ have some musical structure, and sound
piano-like.  Is it possible to add an additional, explicit penalty for overly
noisy samples?  Would the discriminator benefit from seeing a spectrogram
instead of the raw audio samples?
- It's unclear whether learning progressively higher sampling rates has any
benefit.  A future experiment will compare these results to a vanilla WGAN-GP
implementation for audio.
- Current SOTA for audio generation has moved away from modelling audio as a
real-valued, continuous distribution.  Could generating discrete, rather than
continuous samples help reduce noise?




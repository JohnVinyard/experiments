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

## Future Investigations

- It's unclear whether learning progressively higher sampling rates has any
benefit.  A future experiment will compare these results to a vanilla WGAN-GP
implementation for audio.
- Current SOTA for audio generation has moved away from modelling audio as a
real-valued, continuous distribution.  Could generating discrete, rather than
continuous samples help reduce noise?




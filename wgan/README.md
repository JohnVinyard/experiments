# DCGAN with Wasserstein Gradient Penalty Loss for Audio Generation

 This experiment is closely related to the
 [Progressively Growing GANS for Audio Generation](../growing-gan-audio)
 experiment.  It uses the same dataset, but does not adopt the incrementally
 growing network during training.  Instead, it adapts the
 [DCGAN architecture](https://arxiv.org/abs/1511.06434) to a single dimension,
 and uses the popular [Wasserstein (earth-mover's distance) loss along with a
 gradient penalty](https://arxiv.org/abs/1704.00028).

## Experiment Details

While still low quality, this is the **best audio I've produced to date with
a generative adversarial network**.

Code for the experiment is [here](wgan.py).

As with the progressively growing GANs experiment, the audio corpus used is
[Johann Sebastian Bach - Complete Partitas for piano - Vol. I](https://archive.org//details/AOC11B).

While still very noisy, results are more musical and more satisfying than those
produced with the growing GAN.  While the growing GAN generator alternates
between upsampling and convolutional layers, this generator architecture
consists solely of transposed convolutions.

Spectrograms of generated audio, and ogg vorbis files with some generated
samples can be found [here](samples).

## Future Investigations

- While these samples are better than those produced by the progressively growing
GAN, they're still very noisy?  Can we explicitly penalize "noisiness" without
overly biasing the generator, or crippling the network's abillity to learn other,
less "clean" signals?
- Try this with a few other categories of sound:
    - NSynth (exclusively single notes)
    - Speech
    - drum loops
    - pop music

## Samples
<audio src="samples/eff326bb28d342b78ab97a5facdcdda8.ogg" />
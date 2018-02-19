# WaveGAN

This experiment is closely related to the previous experiment
[DCGAN with Wasserstein Gradient Penalty Loss for Audio Generation](../wgan)
experiment.  It uses the same dataset, but is an approximate implementation of
the recent [WaveGAN](https://arxiv.org/abs/1802.04208) paper by Chris Donahue,
Julian McAuley, and Miller Puckette.  Their approach is pretty darn close to the
approach outlined in the previously mentioned experiment.

## Experiment Details

The WaveGAN paper has a few important details, one or more of which is likely
responsible for improved audio quality, and overall sample coherence:

- larger kernels (width 25 at each layer)
- upsampling followed by convolution in the generator, rather than
 transposed convolutions
- bias in all layers (I did not use biases at all in the [DCGAN with Wasserstein Gradient Penalty Loss for Audio Generation](../wgan) experiment)

This experiment does not yet implement/include phase shift in the
critic/discriminator.

Code for the experiment is [here](wave_gan.py).

As with the progressively growing GANs experiment, the audio corpus used is
[Johann Sebastian Bach - Complete Partitas for piano - Vol. I](https://archive.org//details/AOC11B).

## Future Investigations
- Implement phase shift
- Ablation studies to understand the contribution of:
    - large vs small kernels
    - bias in some/all layers
    - unlearned nearest-neighbor upsampling vs transposed convolutions


## Samples
Samples from the model after ~2 days on a GeForce GTX 960 can be heard here:

https://soundcloud.com/user-961608881/sets/wavegan

# GAN with Coarse Up/Down Sampling and Dilations

This experiment is closely related to
[DCGAN with Wasserstein Gradient Penalty Loss for Audio Generation](../wgan).
It uses the same dataset but alters the architecture of the generator and
discriminator to include dilated convolutions.

## Experiment Details

This experiment was a failure, the network learned very slowly, and never
produced results anywhere near the (relatively low) quality of
[this experiment](../wgan).  I have not included audio samples.

The samples were both more noisy and less globally coherent.

Code for the experiment is [here](dilated_gan.py).

The generator and discriminator are mirror images of one another, and have the
following high-level architecture:

### Generator
- Use very "coarse" (large kernels and large strides) transposed convolutions
to quickly scale features maps up to the input sample size (8192 samples, in this
case)
- Apply several layers of increasingly dilated convolutions to this feature
map, maintaining both the feature map size and the number of channels throughout.
- Apply a final 1x1 convolution that transforms features into raw samples

### Discriminator
- Apply several layers of increasingly dilated convolutions to the input sample,
maintaining the feature map size and number of channels throughout
- Use very "coarse" (large kernels and large strides) convolutions to quickly
scale feature map sizes down to 1
- Connect to a final linear layer
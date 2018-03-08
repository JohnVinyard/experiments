# AutoEncoder Trained with Perceptually Inspired Loss Function

This experiment seeks to understand the outcome of using a perceptually-inspired
loss function for an autoencoder that encodes and decodes raw audio.  I compare
results using mean squared error in raw sample space, to mean squared error in
a log frequency and amplitude spectrogram space.

## Experiment Details
Code for the experiment is [here](raw_sample_autoencoder.py).

You can hear some audio samples [here](samples).

The dataset used is [Luca Brasi 2 by Kevin Gates](https://archive.org/details/LucaBrasi2),
as it features a wide range of complex sounds, including human speech, drums, and
synthetic instruments.

All-convolutional (and transposed convolutional) networks are used for the
encoder and decoder respectively.  In other words, all up and down-sampling is
learned.  Both encoder and decoder networks use a gated activation function
in the style of wavenet and pixelCNN.

The goal is to learn an encoding of 2048 sample segments at 11.025Khz (~185ms)
in a much lower-dimensional (128, in this experiment) space that preserves
enough information to produce pleasing results.

### Mean-Squared Error
MSE loss is used as a baseline. The model is trained for 200 epochs on the
dataset, and ultimately produces recognizable audio that suffers from the
following problems:

- fails to reproduce quite a bit of high-frequency content
- is noisy/staticy

### Perceptual Loss
The perceptual loss function first transforms the original and decoded audio by:

1. Passing the audio through a bark-spaced FIR filter bank
1. Applying half-wave rectification (using ReLU)
1. Applying log-scaled amplitude

Then, it computes the mean squared error in this new representation space.  This
results in audio that is ultimately _less_ pleasing than the audio resulting from
the MSE loss function, but the audio is different is some very interesting ways:

- It does a better job of reproducing high-frequency content (likely due to the
 log amplitude scaling across frequency bands)
- speech is less intelligible
- the noise is much more tonal; instead of sounding staticy, reproduced voices
 sound like a vocoder

**The really interesting takeaway is that we can influence the outcome by
tweaking the space in which the loss function is computed**

## Future Investigations

- Try a categorical loss, over discretized raw samples (wavenet and pixelCNN style)
- Try a discretized mixture of logistics loss (pixelCNN++ style)
- Try adding an adversarial loss (i.e., a learned loss function)
- Try computing MSE in the feature space of some pre-trained classifier network


The latter two ideas are drawn from
[Generating Images with Perceptual Similarity Metrics based on Deep Networks](https://arxiv.org/abs/1602.02644).

## Samples
You can hear some audio samples [here](samples).
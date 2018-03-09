# Deep Audio Prior

I'm on the hunt for good generative neural network architectures for raw audio,
so I was pretty interested in trying something analogous to the
[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) paper in
the audio domain.

The paper makes a pretty compelling case that there are good priors built into
the architecture of convolutional neural networks that can be exploited for
common image restoration tasks with little to no learning.

The authors optimize the parameters of a convolutional network to reduce the l2
norm of the difference between the network's output, and a corrupted image.  They
show that while the network will _eventually_ "overfit" and reproduce the
corrupted, noisy image exactly, it generally will first produce a more
aesthetically pleasing, "de-noised" version of the corrupted image in early to
middle iterations.

As most of the architectures I'm using for audio are simply adapted to the
audio domain (2d => 1d) from well-known image models, I naturally wondered if
these results hold for audio.  If they do, I might learn a little bit about which
architecture variants are best.  If they _don't_, then I might need to re-think
the architectures I'm currently using in general.


## Experiment Details
I tried the denoising task on 8192-sample-long audio segments at 11.025Khz
(~750ms), with four different upsampling methods:

- linear
- nearest-neighbor
- discrete cosine transform
- learned (using transposed convolutions)

The network architecture is akin to the decoder part of an autoencoder, or the
generator in a generative adversarial network setting; its input is a noise
vector (frozen, in this case), and it applies alternating upsampling and
convolutional layers until an output of dimension 8192 is reached.

I found that the assumptions about a good prior transfer to the audio domain
_for certain kinds of upsampling_.  Linear, and DCT-based upsampling, while
introducing some of their own artifacts, clearly produced less noisy versions
of the corrupted samples.  Nearest-neighbor and transposed convolution
upsampling both quickly "overfit" the corrupted samples, reproducing the
unwanted noise faithfully.  It seems the authors likely encountered similar
results, given this excerpt from the
[supplementary materials](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer):

> An alternative upsamplingmethod could be to use transposed convolutions,
> but the results we obtained using them were worse.

This is fascinating.  Initially, I expected that _every_ variant would
immediately "overfit" to the noise.  How could a high capacity network not
overfit on a dataset of size one?  Another interesting, and possibly related
finding of the authors is that too many skip connections in the network hurt
image restoration performance.


### Questions
- Do certain kinds of upsampling limit the flow of gradient the same way
  restricting skip connections might?  Is limited gradient flow key to this
  working?
- Do certain kinds of upsampling act as a kind of regularization for the network?

### Notes
- I only tried the de-noising task, and did not try in-painting or super
resolution
- I left out the "encoder" part of the architecture described in the
[supplementary materials](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer)
section, as I'm very skeptical that this contributes in any way.  I tried a few
initial experiments with and without it, and couldn't tell the difference.


## Future Investigations
- try an in-painting experiment, remembering that a mask is applied to the
corrupted region to exclude, heading off an obvious path to "overfitting"
- try super resolution


## Samples
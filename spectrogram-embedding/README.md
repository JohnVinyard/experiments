# Unsupervised Spectrogram Embeddings

In [Unsupervised Learning of Semantic Audio Representations](https://arxiv.org/abs/1711.02209)
the authors describe an approach to learning embeddings of spectrograms in
an unsupervised manner.  They make three very astute observations, two of which
I leverage in this implementation:

- noise, translations in time, and small frequency transpositions do not change
the semantics of the sound
- ~~a mixture of two sounds inherits the categories of the constituents~~
- events in close temporal proximity are likely to be semantically similar

## Experiment Details

Code for the experiment is [here](spectrogram_embedding.py).

I first process ~50 hours of audio for various sources, including:

- [Internet Archive](internetarchive.org)
- [Phat Drum Loops](http://www.phatdrumloops.com/beats.php)
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/)
- [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)
- [FreeSound](https://freesound.org/)

For a single training batch I do the following:

1. choose the **anchor** spectrograms
2. randomly choose one type of "deformation" for the batch from these options:
    - temporal proximity (immediately before or after)
    - additive noise
    - time stretch or contraction
    - pitch transposition
3. Apply the deformation to all anchor spectrograms to compute the **positive** samples
4. Choose another set of spectrograms, completely at random as the **negative** samples
5. Apply the triplet loss to the batch

The aim here is to push the deformed, but semantically similar/identical examples
closer to our anchors in our 128-dimensional embedding space, while pushing the
negative examples further away.

Subjective (and completely solo) listening tests indicate that the network does
in fact learn a useful embedding of the spectrograms.  Perceptually similar
moments (even from totally different audio source files) appear in clusters
together.

## Future Investigations

- Is there a better way to choose negative examples?  It's possible, but
hopefully unlikely, that negative examples are also semantically/perceptually
similar to the anchor.  As long as this happens infrequently, it probably
doesn't matter
- Is vector math possible?  E.g., can I choose the embedding of a drum loop,
and the embedding of a solo guitar sound, and find samples of drums and guitar
playing together?


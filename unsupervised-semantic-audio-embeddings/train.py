from __future__ import division
import torch
from torch.optim import Adam
from torch import nn
from scipy.spatial.distance import cdist
import numpy as np
from time import sleep
import threading


class Trainer(object):
    """
    The driver for the embedding network's training
    """

    def __init__(
            self,
            network,
            triplet_sampler,
            learning_rate,
            triplet_loss_margin=0.1,
            batch_size=32,
            batch_queue_size=20):
        super(Trainer, self).__init__()
        self.batch_queue_size = batch_queue_size
        self.batch_size = batch_size
        self.triplet_loss_margin = triplet_loss_margin
        self.learning_rate = learning_rate
        self.triplet_sampler = triplet_sampler
        self.batch_queue = BatchQueue(
            self.triplet_sampler, self.batch_size, self.batch_queue_size)
        self.network = network
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, network.parameters()),
            lr=learning_rate)
        self.device = torch.device('cpu')
        self.loss = nn.TripletMarginLoss(margin=self.triplet_loss_margin)

    def to(self, device):
        self.device = device
        return self

    def _distance(self, a, b):
        # KLUDGE: The distance computation here assumes that all embeddings
        # already have unit norm
        return 1 - (a * b).sum(axis=-1)

    def _best_negative_indices(
            self, anchors, negatives, anchor_to_positive_distances):
        dist_matrix = cdist(anchors, negatives, metric='cosine')
        diff = dist_matrix - anchor_to_positive_distances[:, None]
        diff[diff <= 0] = np.finfo(diff.dtype).max
        indices = np.argmin(diff, axis=-1)
        return indices

    def negative_mining_demo(self):
        anchors, positives, negatives = \
            self._sample_batch_and_compute_embeddings()

        anchors = anchors.data.cpu().numpy()
        positives = positives.data.cpu().numpy()
        negatives = negatives.data.cpu().numpy()

        anchor_to_positive_distances = self._distance(anchors, positives)
        anchor_to_negative_distances = self._distance(anchors, negatives)

        indices = self._best_negative_indices(
            anchors, negatives, anchor_to_positive_distances)
        new_anchor_to_negative_distances = self._distance(
            anchors, negatives[indices])

        return \
            anchor_to_positive_distances, \
            anchor_to_negative_distances, \
            new_anchor_to_negative_distances

    def negative_mining(self, anchors, positives, negatives):
        """
        Reorganize the batch to make it more "difficult" by assigning negative
        examples to the nearest anchor example, with the additional constraint
        that they not be nearer than the positive example.  In other words,
        mimimize the positive margin between positive and negative examples
        within the batch
        """
        device = anchors.device

        anchors = anchors.data.cpu().numpy()
        positives = positives.data.cpu().numpy()
        negatives = negatives.data.cpu().numpy()

        anchor_to_positive_distances = self._distance(anchors, positives)
        indices = self._best_negative_indices(
            anchors, negatives, anchor_to_positive_distances)
        return torch.from_numpy(indices).to(device)

    def _is_bad_update(self):
        for p in self.network.parameters():
            if p.data.grad is None:
                continue

            if bool(torch.isnan(p.data.grad).sum().item()):
                print('Bad gradients detected!')
                return True

        return False

    def _sample_batch_and_compute_embeddings(self):
        batch = self.batch_queue.pop()
        batch = torch.from_numpy(batch).float().to(self.device)

        anchors = batch[:, 0, :]
        positives = batch[:, 1, :]
        negatives = batch[:, 2, :]

        # compute embeddings for all examples
        anchor_embedding = self.network(anchors)
        positive_embedding = self.network(positives)
        negative_embedding = self.network(negatives)
        return anchor_embedding, positive_embedding, negative_embedding

    def train(self):
        """
        Sample batches, perform within-batch semi-hard negative mining, compute
        the triplet loss and update network weights.
        """
        while True:
            self.network.zero_grad()

            anchor_embedding, positive_embedding, negative_embedding = \
                self._sample_batch_and_compute_embeddings()

            # reorder negative examples to maximize triplet difficulty
            hard_negative_indices = self.negative_mining(
                anchor_embedding, positive_embedding, negative_embedding)
            hard_negatives = negative_embedding[hard_negative_indices]

            # compute the triplet loss
            error = self.loss(
                anchor_embedding, positive_embedding, hard_negatives)

            # compute the gradients
            error.backward()

            if self._is_bad_update():
                # don't update the weights with these gradients, as they
                # contain bad values
                continue

            # update the network weights
            self.optimizer.step()
            yield error.item()


class BatchQueue(object):
    """
    Since sampling batches involves some IO and computation on the CPU, sample
    batches in a thread separate from the main program so that the queue of
    batches for training stays full (hopefully) and the GPU does not sit idle.
    """

    def __init__(self, triplet_sampler, batch_size, queue_size):
        super(BatchQueue, self).__init__()
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.triplet_sampler = triplet_sampler
        self.queue = []
        self.t = threading.Thread(target=self.run)
        self.t.daemon = True
        self.t.start()

    def run(self):
        """
        Sample batches indefinitely, ensuring that our queue is always of size
        self.queue_size
        """
        while True:
            if len(self.queue) < self.queue_size:
                self.queue.append(self.triplet_sampler.sample(self.batch_size))
            sleep(0.1)

    def pop(self):
        """
        Grab a sample from the queue, blocking until one is available
        """
        while len(self.queue) < self.queue_size:
            continue
        return self.queue.pop()

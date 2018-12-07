import torch
from torch.optim import Adam
from torch import nn
from scipy.spatial.distance import cdist
import numpy as np
from time import sleep
import threading


class BatchQueue(object):
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
        while True:
            if len(self.queue) < self.queue_size:
                self.queue.append(self.triplet_sampler.sample(self.batch_size))
            sleep(0.1)

    def pop(self):
        while len(self.queue) < self.queue_size:
            continue
        return self.queue.pop()


class Trainer(object):
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

    def negative_mining(self, anchors, positives, negatives):
        device = anchors.device

        anchors = anchors.data.cpu().numpy()
        positives = positives.data.cpu().numpy()
        negatives = negatives.data.cpu().numpy()

        anchor_to_positive_distances = \
            np.linalg.norm(anchors - positives, axis=-1)
        dist_matrix = cdist(anchors, negatives, metric='cosine')
        diff = dist_matrix - anchor_to_positive_distances[:, None]
        diff[diff <= 0] = np.finfo(diff.dtype).max
        indices = np.argmin(diff, axis=-1)
        return torch.from_numpy(indices).to(device)

    def train(self):
        while True:
            batch = self.batch_queue.pop()
            batch = torch.from_numpy(batch).float().to(self.device)

            anchors = batch[:, 0, :]
            positives = batch[:, 1, :]
            negatives = batch[:, 2, :]

            # compute embeddings for all examples
            anchor_embedding = self.network(anchors)
            positive_embedding = self.network(positives)
            negative_embedding = self.network(negatives)

            # reorder negative examples to maximize triplet difficulty
            hard_negative_indices = self.negative_mining(
                anchor_embedding, positive_embedding, negative_embedding)
            hard_negatives = negative_embedding[hard_negative_indices]

            # compute the triplet loss
            error = self.loss(
                anchor_embedding, positive_embedding, hard_negatives)

            # compute the gradients
            error.backward()

            # update the network weights
            self.optimizer.step()
            yield error.item()



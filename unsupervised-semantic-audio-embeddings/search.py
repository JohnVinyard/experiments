from __future__ import division, print_function
from scipy.spatial.distance import cdist
import heapq
import numpy as np
from random import choice
import zounds
from time import time
from itertools import product


def batch_unit_norm(b, epsilon=1e-8):
    """
    Give all vectors unit norm along the last dimension
    """
    return b / np.linalg.norm(b, axis=-1, keepdims=True) + epsilon


def unit_vectors(n_examples, n_dims):
    """
    Create n_examples of synthetic data on the unit
    sphere in n_dims
    """
    dense = np.random.normal(0, 1, (n_examples, n_dims))
    return batch_unit_norm(dense)


def hyperplanes(n_planes, n_dims):
    """
    Return n_planes plane vectors, which describe
    hyperplanes in n_dims space that are perpendicular
    to lines running from the origin to each point
    """
    return unit_vectors(n_planes, n_dims)


def random_projection(plane_vectors, data, pack=True, binarize=True):
    """
    Return bit strings for a batch of vectors, with each
    bit representing which side of each hyperplane the point
    falls on
    """

    flattened = data.reshape((len(data), plane_vectors.shape[-1]))
    x = np.dot(plane_vectors, flattened.T).T
    if not binarize:
        return x

    output = np.zeros((len(data), len(plane_vectors)), dtype=np.uint8)
    output[np.where(x > 0)] = 1

    if pack:
        output = np.packbits(output, axis=-1).view(np.uint64)

    return output


class HyperPlaneNode(object):
    def __init__(self, shape, data=None):
        super(HyperPlaneNode, self).__init__()
        self.dimensions = shape

        # choose one plane, at random, for this node
        self.plane = hyperplanes(1, shape)

        self.data = \
            data if data is not None else np.zeros((0,), dtype=np.uint64)

        self.left = None
        self.right = None

    def __len__(self):
        return len(self.data)

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def children(self):
        return self.left, self.right

    def distance(self, query):
        dist = random_projection(
            self.plane, query, pack=False, binarize=False).reshape(-1)
        return dist

    def route(self, data, indices=None):

        if indices is None:
            indices = self.data
        data = data[indices]

        dist = self.distance(data)
        left_indices = indices[dist > 0]
        right_indices = indices[dist <= 0]
        return left_indices, right_indices

    def create_children(self, data):
        left_indices, right_indices = self.route(data)
        self.left = HyperPlaneNode(self.dimensions, left_indices)
        self.right = HyperPlaneNode(self.dimensions, right_indices)


class MultiHyperPlaneTree(object):
    def __init__(self, data, smallest_node, n_trees=10):
        super(MultiHyperPlaneTree, self).__init__()
        self.dimensions = data.shape[1]
        self.data = data
        indices = np.arange(0, len(data), dtype=np.uint64)
        self.smallest_node = smallest_node

        self.roots = \
            [HyperPlaneNode(self.dimensions, indices) for _ in xrange(n_trees)]
        build_queue = list(self.roots)

        while build_queue:
            node = build_queue.pop()

            if len(node) <= smallest_node:
                continue
            else:
                node.create_children(self.data)
                build_queue.extend(node.children)

    def append(self, chunk):

        # compute the new set of indices that need to be added to the tree
        new_indices = np.arange(0, len(chunk), dtype=np.uint64) + len(self.data)

        # ensure that the chunk of vectors are added to the available vector
        # data
        self.data = np.concatenate([self.data, chunk])

        # initialize the search queue with all root nodes
        search_queue = list([(r, new_indices) for r in self.roots])

        while search_queue:

            # add the indices to the node's data
            node, indices = search_queue.pop()

            node.data = np.concatenate([node.data, indices])

            if len(node) <= self.smallest_node:
                # this will be a leaf node.  There's no need to further route
                # the data or add further child nodes (for now)
                continue

            if node.is_leaf:
                # we'll be creating new child nodes.  At this point, we need
                # to route *all* of the data currently owned by this node
                node.create_children(self.data)
            else:
                # this node already has children, so it's only necessary to
                # route new indices
                left_indices, right_indices = node.route(self.data, indices)
                search_queue.append((node.left, left_indices))
                search_queue.append((node.right, right_indices))

    def search_with_priority_queue(self, query, n_results, threshold):
        query = query.reshape(1, self.dimensions)

        indices = set()

        # this is kinda arbitrary.
        # How do I pick this intelligently?
        to_consider = n_results * 100

        # put the root nodes in the queue
        heap = [(-9999, root) for root in self.roots]

        # traverse the tree, finding candidate indices
        while heap and len(indices) < to_consider:
            current_distance, current_node = heapq.heappop(heap)

            if current_node.is_leaf:
                indices.update(current_node.data)
                continue

            dist = current_node.distance(query)
            abs_dist = np.abs(dist)
            below_threshold = abs_dist < threshold

            if dist > 0 or below_threshold:
                heapq.heappush(heap, (-abs_dist, current_node.left))

            if dist <= 0 or below_threshold:
                heapq.heappush(heap, (-abs_dist, current_node.right))

        # perform a brute-force distance search over a subset of the data
        indices = np.array(list(indices), dtype=np.uint64)
        data = self.data[indices]
        dist = cdist(query, data, metric='cosine').squeeze()
        partitioned_indices = np.argpartition(dist, n_results)[:n_results]
        sorted_indices = np.argsort(dist[partitioned_indices])
        srt_indices = partitioned_indices[sorted_indices]
        return indices[srt_indices]


class TreeSearch(object):
    def __init__(self, brute_force_search, nodes_per_tree=1024, n_trees=32):
        super(TreeSearch, self).__init__()
        self.brute_force_search = brute_force_search
        self.tree_search = MultiHyperPlaneTree(
            self.brute_force_search.index, nodes_per_tree, n_trees)

    def _brute_force(self, query, nresults):
        distances = cdist(
            query[None, ...],
            self.brute_force_search.index,
            metric=self.brute_force_search.distance_metric)
        return np.argsort(distances[0])[:nresults]

    def _tree(self, query, nresults, tolerance=0.01):
        return self.tree_search.search_with_priority_queue(
            query, nresults, tolerance)

    def random_search(self, n_results=50):
        query = choice(self.brute_force_search.index)
        indices = self._tree(query, n_results)
        return zounds.index.SearchResults(
            query, (self.brute_force_search._ids[i] for i in indices))

    def compare(self, n_results=50, tolerance=0.01):
        bfs_times = []
        tree_times = []
        overlaps = []

        for i in xrange(10):
            query = choice(self.brute_force_search.index)

            start = time()
            brute_force_indices = self._brute_force(query, n_results)
            bfs_times.append(time() - start)

            start = time()
            tree_indices = self._tree(query, n_results, tolerance=tolerance)
            tree_times.append(time() - start)

            intersection = set(brute_force_indices) & set(tree_indices)
            overlap = len(intersection) / len(brute_force_indices)
            overlaps.append(overlap)

        return \
            sum(overlaps) / len(overlaps), \
            sum(bfs_times) / len(bfs_times), \
            sum(tree_times) / len(tree_times)


def experiment(search):
    n_trees = [32]
    tolerances = [0.001]
    nodes_per_tree = [1024]

    for nt, t, nodes in product(n_trees, tolerances, nodes_per_tree):
        tree_search = TreeSearch(search, nodes, nt)
        overlap, bfs_time, tree_time = \
            tree_search.compare(n_results=50, tolerance=t)
        results = dict(
            n_trees=nt,
            tolerance=t,
            nodes_per_tree=nodes,
            overlap=overlap,
            bfs_time=bfs_time,
            tree_time=tree_time)
        print(results)

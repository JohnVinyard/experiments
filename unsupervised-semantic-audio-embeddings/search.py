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


def random_projection(plane_vectors, data):
    """
    Take the dot product of data with a set of points that define hyperplanes
    in that they lie perpendicular to the plane.  The sign of each output
    dimension indicates the side of the plane on which each data point lies.
    """
    flattened = data.reshape((len(data), plane_vectors.shape[-1]))
    return np.dot(plane_vectors, flattened.T).T


class HyperPlaneNode(object):
    """
    A single node in an annoy-like (https://github.com/spotify/annoy) index
    which subdivides data by defining a hyperplane and assigning to child nodes
    based on which side of the hyperplane each data point lies
    """

    def __init__(self, shape, data=None):
        super(HyperPlaneNode, self).__init__()
        self.dimensions = shape

        # define a hyperplane with a point that lies perpendicular to the plane
        self.plane = unit_vectors(1, shape)

        self.data = \
            data if data is not None else np.zeros((0,), dtype=np.uint64)

        self.left = None
        self.right = None

    def __len__(self):
        """
        The number of items held by this node and all descendant nodes
        """
        return len(self.data)

    @property
    def is_leaf(self):
        """
        True when the node has no children
        """
        return self.left is None and self.right is None

    @property
    def children(self):
        """
        Returns a two-tuple of this node's left and right sub-nodes
        """
        return self.left, self.right

    def distance(self, query):
        """
        Compute the distance from this node's hyperplane, the sign of which
        determines the *side* of the hyperplane on which each data point falls
        """
        return random_projection(self.plane, query).reshape(-1)

    def route(self, data, indices=None):
        """
        Return the indices of elements that should be routed to the left and
        right nodes, respectively
        """
        if indices is None:
            indices = self.data
        data = data[indices]

        dist = self.distance(data)
        left_indices = indices[dist > 0]
        right_indices = indices[dist <= 0]
        return left_indices, right_indices

    def create_children(self, data):
        """
        Look at some incoming data, and route it to two new left and right
        child nodes
        """
        left_indices, right_indices = self.route(data)
        self.left = HyperPlaneNode(self.dimensions, left_indices)
        self.right = HyperPlaneNode(self.dimensions, right_indices)


class HyperPlaneTree(object):
    """
    Create a log time index that allows approximate nearest-neighbor searches
    over high-dimensional data by building multiple trees that subdivide data
    at each node by splitting space with a random hyperplane
    """

    def __init__(self, data, smallest_node, n_trees=10):
        super(HyperPlaneTree, self).__init__()
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
        """
        Append a new chunk of data to the index, assigning it to the correct
        nodes, and creating new nodes if current leaf nodes have grown beyond
        the configured size.
        """

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

    def _brute_force_search(self, indices, query, n_results):
        """
        Perform a brute-force search over a subset of the data, narrowed
        significantly using the index.
        """
        indices = np.array(list(indices), dtype=np.uint64)
        data = self.data[indices]
        dist = cdist(query, data, metric='cosine').squeeze()
        partitioned_indices = np.argpartition(dist, n_results)[:n_results]
        sorted_indices = np.argsort(dist[partitioned_indices])
        srt_indices = partitioned_indices[sorted_indices]
        return indices[srt_indices]

    def search(self, query, n_results, threshold, n_trees=None):
        """
        Perform an approximate nearest-neighbors search to find n_results that
        are similar (have a low cosine distance or angle) with query
        """
        query = query.reshape(1, self.dimensions)

        # indices of candidate elements that will be further narrowed and
        # ordered using a brute force search once all trees have been searched
        indices = set()

        # TODO: this number of candidate results to consider is entirely
        # arbitrary.  How do I choose this in a principled way?
        to_consider = n_results * 100

        roots = self.roots[:n_trees]
        # put the root nodes in the queue
        heap = [(None, root) for root in roots]

        # traverse the tree, finding candidate indices
        while heap and len(indices) < to_consider:
            current_distance, current_node = heapq.heappop(heap)

            if current_node.is_leaf:
                indices.update(current_node.data)
                continue

            dist = current_node.distance(query)
            abs_dist = np.abs(dist)
            below_threshold = abs_dist < threshold

            # route the query to the appropriate next node and push it onto
            # the priority heap, assigning its priority in proportion to its
            # distance to the hyperplane.  Hyperplanes that are further away
            # will be a better split for this query, while nearby hyperplanes
            # may subdivide space in an undesirable way.  If the hyperplane is
            # too near, simply search both paths.
            if dist > 0 or below_threshold:
                heapq.heappush(heap, (-abs_dist, current_node.left))

            if dist <= 0 or below_threshold:
                heapq.heappush(heap, (-abs_dist, current_node.right))

        return self._brute_force_search(indices, query, n_results)


class TreeSearch(object):
    """
    Build an annoy-like log time approximate nearest neighbors search from the
    data stored in a brute-force search "index", which contains every embedding
    in our database
    """
    def __init__(self, brute_force_search, nodes_per_tree=1024, n_trees=32):
        super(TreeSearch, self).__init__()
        self.brute_force_search = brute_force_search
        self.tree_search = HyperPlaneTree(
            self.brute_force_search.index, nodes_per_tree, n_trees)

    def _brute_force(self, query, nresults):
        distances = cdist(
            query[None, ...],
            self.brute_force_search.index,
            metric=self.brute_force_search.distance_metric)
        return np.argsort(distances[0])[:nresults]

    def _tree(self, query, nresults, tolerance=0.01, n_trees=None):
        return self.tree_search.search(
            query, nresults, tolerance, n_trees=n_trees)

    def random_search(self, n_results=50):
        query = choice(self.brute_force_search.index)
        indices = self._tree(query, n_results)
        return zounds.index.SearchResults(
            query, (self.brute_force_search._ids[i] for i in indices))

    def compare(
            self, n_results=50, tolerance=0.001, n_trees=None, n_iterations=10):

        bfs_times = []
        tree_times = []
        overlaps = []

        for i in xrange(n_iterations):
            query = choice(self.brute_force_search.index)

            start = time()
            brute_force_indices = self._brute_force(query, n_results)
            bfs_times.append(time() - start)

            start = time()
            tree_indices = self._tree(
                query, n_results, tolerance=tolerance, n_trees=n_trees)
            tree_times.append(time() - start)

            intersection = set(brute_force_indices) & set(tree_indices)
            overlap = len(intersection) / len(brute_force_indices)
            overlaps.append(overlap)

        return \
            sum(overlaps) / len(overlaps), \
            sum(bfs_times) / len(bfs_times), \
            sum(tree_times) / len(tree_times)

    def visualize_tree(self):
        from graphviz import Graph
        g = Graph('G', filename='tree.gv', format='svg')
        g.attr(
            'node',
            label=str(1),
            shape='square',
            style='filled',
            color='black',
            width=str(0),
            height=str(0))
        g.attr('edge', penwidth=str(3))

        def node_name(node):
            return str(hash(node))

        root_node = self.tree_search.roots[0]
        stack = [root_node]
        g.node(node_name(root_node))

        while stack:
            node = stack.pop()
            if not node.is_leaf:
                name = node_name(node)
                left_name = node_name(node.left)
                right_name = node_name(node.right)
                g.node(left_name)
                g.node(right_name)
                g.edge(name, left_name)
                g.edge(name, right_name)
                stack.append(node.left)
                stack.append(node.right)
        g.view()

    def compare_and_plot(self, n_trees, n_iterations=100, n_results=100):
        from matplotlib import pyplot as plt

        overlaps = []
        bf_times = []
        tree_times = []

        for nt in n_trees:
            overlap, bf_time, tree_time = self.compare(
                n_trees=nt, n_iterations=n_iterations, n_results=n_results)
            overlaps.append(overlap)
            bf_times.append(bf_time)
            tree_times.append(tree_time)
            print(nt, overlap, bf_time, tree_time)

        bf_times = 1.0 / np.array(bf_times)
        tree_times = 1.0 / np.array(tree_times)

        fig = plt.figure()

        plt.scatter(
            [1.0],
            [np.array(bf_times).mean()],
            c='red',
            marker='s',
            label='brute_force')

        plt.plot(
            overlaps,
            tree_times,
            marker='x',
            label='hyperplane_tree')
        for nt, overlap, tree_time in zip(n_trees, overlaps, tree_times):
            plt.annotate(nt, xy=[overlap, tree_time], textcoords='data')

        plt.xlabel('accuracy', fontsize=20)
        plt.ylabel('queries per second', fontsize=20)
        plt.yscale('log', basey=10)
        plt.legend()
        plt.savefig('search_times.png', format='png')
        fig.clf()


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

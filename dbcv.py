"""
Implimentation of Density-Based Clustering Validation "DBCV"
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph
from tqdm import tqdm


class DBCV:
    def __init__(self, samples: np.ndarray, labels: np.ndarray, dist_function: str = 'euclidean', verbose=False):
        """
        Density Based clustering validation

        Args:
            samples (np.ndarray): ndarray with dimensions [n_samples, n_features]
                data to check validity of clustering
            labels (np.array): clustering assignments for data X
            dist_dunction (func): function to determine distance between objects
                func args must be [np.array, np.array] where each array is a point
        """
        self.samples = samples
        self.labels = labels
        self.dist_function = dist_function
        self.cluster_lookup = {}
        self.shortest_paths = None
        self.verbose = verbose

    def verbose_log(self, msg):
        if self.verbose:
            print(msg)

    def get_score(self):
        """
        Density Based clustering validation

        Returns: cluster_validity (float)
            score in range[-1, 1] indicating validity of clustering assignments
        """
        graph = self._mutual_reach_dist_graph(self.samples, self.labels, self.dist_function)
        self.verbose_log("made graph matrix")
        mst = self._mutual_reach_dist_MST(graph)
        self.verbose_log("built MST")
        self.shortest_paths = csgraph.dijkstra(mst)
        self.verbose_log("calculated shortest paths")
        cluster_validity = self._clustering_validity_index(mst, self.labels)
        self.verbose_log("scores calculated")
        return cluster_validity

    def _core_dist(self, point: np.ndarray, distance_vector: np.ndarray):
        """
        Computes the core distance of a point.
        Core distance is the inverse density of an object.

        Args:
            point (np.array): array of dimensions (n_features,)
                point to compute core distance of

            distance_vector (np.array):
                vector of distances from point to all other points in its cluster

        Returns: core_dist (float)
            inverse density of point
        """
        n_features = np.shape(point)[0]
        n_neighbors = np.shape(distance_vector)[0]

        distance_vector = distance_vector[distance_vector != 0]
        numerator = ((1 / distance_vector) ** n_features).sum()
        core_dist = (numerator / (n_neighbors - 1)) ** (-1 / n_features)
        return core_dist

    def _calculate_pairwise_distance(self, samples: np.ndarray, dist_function: str):
        # TODO: align the metric with distance function
        return cdist(samples, samples, metric=dist_function)

    def _mutual_reach_dist_graph(self, X, labels, dist_function):
        """
        Computes the mutual reach distance complete graph.
        Graph of all pair-wise mutual reachability distances between points

        Args:
            X (np.ndarray): ndarray with dimensions [n_samples, n_features]
                data to check validity of clustering
            labels (np.array): clustering assignments for data X
            dist_dunction (func): function to determine distance between objects
                func args must be [np.array, np.array] where each array is a point

        Returns: graph (np.ndarray)
            array of dimensions (n_samples, n_samples)
            Graph of all pair-wise mutual reachability distances between points.

        """
        n_samples = np.shape(X)[0]

        pairwise_distance = self._calculate_pairwise_distance(X, dist_function)
        core_dists = []

        for idx in tqdm(range(n_samples)):
            class_label = labels[idx]
            members = self._get_label_member_indices(labels, class_label)
            distance_vector = pairwise_distance[idx, :][members]
            core_dists.append(self._core_dist(X[idx], distance_vector))

        # to do a bulk np.max we want to repeat core distances
        core_dists = np.repeat(np.array(core_dists).reshape(-1, 1), n_samples, axis=1)

        # this matrix and its inverse show core_dist in position i,j for point i and point j respectively
        core_dists_i = core_dists[:, :, np.newaxis]
        core_dists_j = core_dists.T[:, :, np.newaxis]
        pairwise_distance = pairwise_distance[:, :, np.newaxis]

        # concatenate all distances to compare them all in numpy
        mutual_reachability_distance_matrix = np.concatenate([core_dists_i, core_dists_j, pairwise_distance], axis=-1)
        graph = np.max(mutual_reachability_distance_matrix, axis=-1)

        return graph

    def _mutual_reach_dist_MST(self, dist_tree):
        """
        Computes minimum spanning tree of the mutual reach distance complete graph

        Args:
            dist_tree (np.ndarray): array of dimensions (n_samples, n_samples)
                Graph of all pair-wise mutual reachability distances
                between points.

        Returns: minimum_spanning_tree (np.ndarray)
            array of dimensions (n_samples, n_samples)
            minimum spanning tree of all pair-wise mutual reachability
                distances between points.
        """
        mst = minimum_spanning_tree(dist_tree).toarray()
        return mst + np.transpose(mst)

    def _cluster_density_sparseness(self, MST, labels, cluster):
        """
        Computes the cluster density sparseness, the minimum density
            within a cluster

        Args:
            MST (np.ndarray): minimum spanning tree of all pair-wise
                mutual reachability distances between points.
            labels (np.array): clustering assignments for data X
            cluster (int): cluster of interest

        Returns: cluster_density_sparseness (float)
            value corresponding to the minimum density within a cluster
        """
        indices = np.where(labels == cluster)[0]
        cluster_MST = MST[indices][:, indices]
        cluster_density_sparseness = np.max(cluster_MST)
        return cluster_density_sparseness

    def _cluster_density_separation(self, MST, labels, cluster_i, cluster_j):
        """
        Computes the density separation between two clusters, the maximum
            density between clusters.

        Args:
            MST (np.ndarray): minimum spanning tree of all pair-wise
                mutual reachability distances between points.
            labels (np.array): clustering assignments for data X
            cluster_i (int): cluster i of interest
            cluster_j (int): cluster j of interest

        Returns: density_separation (float):
            value corresponding to the maximum density between clusters
        """
        indices_i = np.where(labels == cluster_i)[0]
        indices_j = np.where(labels == cluster_j)[0]

        relevant_paths = self.shortest_paths[indices_i][:, indices_j]
        density_separation = np.min(relevant_paths)
        return density_separation

    def _cluster_validity_index(self, MST, labels, cluster):
        """
        Computes the validity of a cluster (validity of assignmnets)

        Args:
            MST (np.ndarray): minimum spanning tree of all pair-wise
                mutual reachability distances between points.
            labels (np.array): clustering assignments for data X
            cluster (int): cluster of interest

        Returns: cluster_validity (float)
            value corresponding to the validity of cluster assignments
        """
        min_density_separation = np.inf
        for cluster_j in np.unique(labels):
            if cluster_j != cluster:
                cluster_density_separation = self._cluster_density_separation(MST,
                                                                              labels,
                                                                              cluster,
                                                                              cluster_j)
                if cluster_density_separation < min_density_separation:
                    min_density_separation = cluster_density_separation
        cluster_density_sparseness = self._cluster_density_sparseness(MST,
                                                                      labels,
                                                                      cluster)
        numerator = min_density_separation - cluster_density_sparseness
        denominator = np.max([min_density_separation, cluster_density_sparseness])
        cluster_validity = numerator / denominator
        return cluster_validity

    def _clustering_validity_index(self, MST, labels):
        """
        Computes the validity of all clustering assignments for a
        clustering algorithm

        Args:
            MST (np.ndarray): minimum spanning tree of all pair-wise
                mutual reachability distances between points.
            labels (np.array): clustering assignments for data X

        Returns: validity_index (float):
            score in range[-1, 1] indicating validity of clustering assignments
        """
        n_samples = len(labels)
        validity_index = 0
        for label in np.unique(labels):
            fraction = np.sum(labels == label) / float(n_samples)
            cluster_validity = self._cluster_validity_index(MST, labels, label)
            validity_index += fraction * cluster_validity
        return validity_index

    def _get_label_member_indices(self, labels, cluster):
        """
        Helper function to get samples of a specified cluster.

        Args:
            labels (np.array): clustering assignments for data X
            cluster (int): cluster of interest

        Returns: members (np.ndarray)
            array of dimensions (n_samples,) of indices of samples of cluster
        """
        if cluster in self.cluster_lookup:
            return self.cluster_lookup[cluster]

        indices = np.where(labels == cluster)[0]

        self.cluster_lookup[cluster] = indices
        return indices


def get_score(samples: np.ndarray, labels: np.ndarray, dist_function: str = 'euclidean', verbose=False):
    scorer = DBCV(samples, labels, dist_function, verbose=verbose)
    return scorer.get_score()

"""
Implimentation of Density-Based Clustering Validation "DBCV"

Citation:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.
"""

import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph
from typing import List, Callable


class DBCV:
    def __init__(self, samples: np.ndarray, labels: np.ndarray, dist_function: Callable = euclidean):
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

    def get_score(self):
        """
        Density Based clustering validation

        Returns: cluster_validity (float)
            score in range[-1, 1] indicating validity of clustering assignments
        """
        graph = self._mutual_reach_dist_graph(self.samples, self.labels, self.dist_function)
        mst = self._mutual_reach_dist_MST(graph)
        cluster_validity = self._clustering_validity_index(mst, self.labels)
        return cluster_validity

    def _core_dist(self, point, distance_vector):
        """
        Computes the core distance of a point.
        Core distance is the inverse density of an object.

        Args:
            point (np.array): array of dimensions (n_features,)
                point to compute core distance of
            neighbors (np.ndarray): array of dimensions (n_neighbors, n_features):
                array of all other points in object class
            dist_dunction (func): function to determine distance between objects
                func args must be [np.array, np.array] where each array is a point

        Returns: core_dist (float)
            inverse density of point
        """
        n_features = np.shape(point)[0]
        n_neighbors = np.shape(distance_vector)[0]

        distance_vector = distance_vector[distance_vector != 0]
        numerator = ((1 / distance_vector) ** n_features).sum()
        core_dist = (numerator / (n_neighbors - 1)) ** (-1 / n_features)
        return core_dist

    def _calculate_pairwise_distance(self, samples: np.ndarray):
        # TODO: align the metric with distance function
        return cdist(samples, samples, metric='euclidean')

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
        graph = []
        counter = 0

        pairwise_distance = self._calculate_pairwise_distance(X)
        core_dist_map = {}

        for row in range(n_samples):
            graph_row = []
            for col in range(n_samples):
                point_i = X[row]
                point_j = X[col]
                class_i = labels[row]
                class_j = labels[col]

                # TODO: use a lookup table for this method
                members_i = self._get_label_member_indices(labels, class_i)
                members_j = self._get_label_member_indices(labels, class_j)

                distance_vector_i = pairwise_distance[row, :][members_i]
                distance_vector_j = pairwise_distance[col, :][members_j]

                if row not in core_dist_map:
                    core_dist_map[row] = self._core_dist(point_i, distance_vector_i)

                if col not in core_dist_map:
                    core_dist_map[col] = self._core_dist(point_i, distance_vector_j)

                distance = pairwise_distance[row, col]
                mutual_reachability_distance = np.max([core_dist_map[row], core_dist_map[col], distance])

                graph_row.append(mutual_reachability_distance)
            counter += 1
            graph.append(graph_row)
        graph = np.array(graph)
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
        shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
        relevant_paths = shortest_paths[:, indices_j]
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
            X (np.ndarray): ndarray with dimensions [n_samples, n_features]
                data to check validity of clustering
            labels (np.array): clustering assignments for data X
            cluster (int): cluster of interest

        Returns: members (np.ndarray)
            array of dimensions (n_samples, n_features) of samples of the
            specified cluster.
        """
        if cluster in self.cluster_lookup:
            return self.cluster_lookup[cluster]

        indices = np.where(labels == cluster)[0]

        self.cluster_lookup[cluster] = indices
        return indices

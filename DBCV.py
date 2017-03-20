import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph

def DBCV(X, labels, dist_function=euclidean):
    """
    Density Based clustering validation
    """
    graph = _mutual_reach_dist_graph(X, labels, dist_function)
    mst = _mutual_reach_dist_MST(graph)
    cluster_validity = _clustering_validity_index(mst, labels)
    return cluster_validity

def _core_dist(point, neighbors, dist_function):
    """
    Computes the core distance of a point
    """
    n_features = np.shape(point)[0]
    n_neighbors = np.shape(neighbors)[1]
    
    numerator = 0
    for row in neighbors:
        numerator += (1/dist_function(point, row))**n_features
    core_dist = (numerator / (n_neighbors)) ** (1/n_features)
    return core_dist
    
def _mutual_reachability_dist(point_i, point_j, neighbors_i, 
                              neighbors_j, dist_function):
    """
    Computes the mutual reachability distance between points
    """
    core_dist_i = _core_dist(point_i, neighbors_i, dist_function)
    core_dist_j = _core_dist(point_j, neighbors_j, dist_function)
    dist = dist_function(point_i, point_j)
    return np.max([core_dist_i, core_dist_j, dist])
    
def _mutual_reach_dist_graph(X, labels, dist_function):
    """
    Computes the mutual reach distance complete graph
    """
    n_samples = np.shape(X)[0]
    graph = np.ones((n_samples, n_samples))
    for row in range(n_samples):
        for col in range(n_samples):
            point_i = X[row]
            point_j = X[col]
            class_i = labels[row]
            class_j = labels[col]
            members_i = _get_label_members(X, labels, class_i)
            members_j = _get_label_members(X, labels, class_j)
            graph[row, col] = _mutual_reachability_dist(point_i, point_j, 
                                                        members_i, members_j, 
                                                        dist_function)
    return graph

def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the 
    mutual reach distance complete graph
    """
    return minimum_spanning_tree(dist_tree).toarray()
    
def _cluster_density_sparseness(MST, labels, cluster):
    """
    Computes the cluster density sparseness
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[indices][:,indices]
    return np.max(cluster_MST)
    
def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Computes the density separation between two clusters
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    shortest_paths = csgraph.dijkstra(b, indices = indice_i)
    relevant_paths = shortest_paths[:,indices_j]
    shortest_path = np.min(relevant_paths)
    return shortest_path
    
def _cluster_validity_index(MST, labels, cluster):
    min_density_separation = np.inf
    for cluster_j in np.unique(labels):
        if cluster_j != cluster:
            cluster_density_separation = _cluster_density_separation(MST, labels, cluster, cluster_j)
            if cluster_density_separation > min_density_separation:
                min_density_separation = cluster_density_separation
    cluster_density_sparseness = _cluster_density_sparseness(MST, labels, cluster)
    numerator = min_density_separation - cluster_density_sparseness
    denominator = np.max([min_density_separation, cluster_density_sparseness])
    return numerator/denominator
    
def _clustering_validity_index(MST, labels):
    n_samples = len(labels)
    validity_index = 0
    for label in np.unique(labels):
        fraction = np.sum(labels==label) / float(n_samples)
        cluster_validty = _cluster_validity_index(MST, labels, cluster)
        validity_index += fraction * cluster_validity
    return validity_index

def _get_label_members(X, labels, cluster):
    indices = np.where(labels == cluster)[0]
    members = X[indices]
    return members

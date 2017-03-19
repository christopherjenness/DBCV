import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse.csgraph import minimum_spanning_tree

def DBCV(X, labels):
    """
    Density Based clustering validation
    """
    return

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
    print(core_dist_i, core_dist_j, dist)
    return np.max([core_dist_i, core_dist_j, dist])
    
def _mutual_reach_dist_graph():
    """
    Computes the mutual reach distance complete graph
    """
    return

def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the 
    mutual reach distance complete graph
    """
    return minimum_spanning_tree(dist_tree).toarray()


X = np.random.rand(3,3)
b = minimum_spanning_tree(X)
b.toarray()
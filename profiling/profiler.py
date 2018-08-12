import DBCV 
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


def generate_data(n_samples=300, noise=0.05):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
    X = noisy_moons[0]
    return X


def generate_labels(X):
    kmeans =  KMeans(n_clusters=2)
    kmeans_labels = kmeans.fit_predict(X)
    return kmeans_labels

if __name__ == '__main__':
    X = generate_data()
    labels = generate_labels(X)
    score = DBCV.DBCV(X, labels, dist_function=euclidean)
    

from dbcv import DBCV
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


def generate_data(n_samples=10000, noise=0.05):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
    X = noisy_moons[0]
    return X


def generate_labels(X):
    kmeans = KMeans(n_clusters=2)
    kmeans_labels = kmeans.fit_predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
    # plt.show()
    return kmeans_labels


if __name__ == '__main__':
    X = generate_data()
    labels = generate_labels(X)
    dbcv = DBCV(X, labels, euclidean)
    score = dbcv.get_score()

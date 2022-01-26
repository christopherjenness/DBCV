import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dbcv import DBCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean



np.random.seed(1)


def get_data():
    noisy_points = np.random.rand(100, 2) - 0.5
    all_points = [noisy_points]

    n_clusters = 4
    cluster_size = 50

    for _ in range(n_clusters):
        cluster_center = np.random.rand(1, 2) - 0.5
        points_x = np.random.normal(loc=cluster_center[0][0], scale=0.03, size=cluster_size)
        points_y = np.random.normal(loc=cluster_center[0][1], scale=0.03, size=cluster_size)

        points = np.array(list(zip(points_x, points_y)))
        all_points.append(points)

    for sop in all_points:
        plt.scatter(sop[:, 0], sop[:, 1])

    plt.show()

    return all_points


def cluster(points):

    samples = np.concatenate(points, axis=0)
    print(samples.shape)

    plt.figure(figsize=(20, 20))

    for i, eps in enumerate([0.01, 0.05, 0.1, 0.2]):
        for j, min_samples in enumerate([5, 20, 50, 100]):

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(samples)
            cnt = Counter()
            cnt.update(clustering.labels_)

            if len(cnt) < 2:
                score = -1
            else:
                sil_score = silhouette_score(samples, clustering.labels_)
                dbcv = DBCV(samples, clustering.labels_, dist_function=euclidean)
                score = dbcv.get_score()

            print(f"{eps}-{min_samples}:\ndbcv-score={score:.2f}\nsil-score:{sil_score:.2f}")

            ax = plt.subplot(4, 4, i * 4 + j + 1)
            plt.scatter(samples[:, 0], samples[:, 1], c=clustering.labels_)
            ax.set_title(f"{eps}-{min_samples}:\ndbcv-score={score:.2f}\nsil-score:{sil_score:.2f}")

    plt.show()


if __name__ == '__main__':
    points = get_data()
    cluster(points)
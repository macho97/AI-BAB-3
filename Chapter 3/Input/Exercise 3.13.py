# Exercise 3.13: K-means clustering using make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=0)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="X", s=180, label="Centers")
plt.title("K-means Clustering with make_blobs")
plt.legend()
plt.show()

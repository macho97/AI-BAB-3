# Example 3.10 Principal Component Analysis Iris
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition

iris = datasets.load_iris()
X, y = iris.data, iris.target

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("sepals length")
plt.ylabel("sepals width")
plt.title("Original Data")

pca = decomposition.PCA(n_components=3)
X1 = pca.fit_transform(X)

plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Data")
plt.show()

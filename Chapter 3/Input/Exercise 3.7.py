# Exercise 3.7: PCA on breast cancer data
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
X = StandardScaler().fit_transform(cancer.data)
X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cancer.target, cmap="coolwarm")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA of Breast Cancer Data")
plt.colorbar(label="Target")
plt.show()

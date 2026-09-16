# Exercise 3.2: SVM using petal length and petal width
from sklearn import datasets, svm

iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Third and fourth features
y = iris.target
clf = svm.SVC()
clf.fit(X, y)
print("Prediction with petal features:", clf.predict([[1.4, 0.2]]))
print("Training accuracy:", clf.score(X, y))

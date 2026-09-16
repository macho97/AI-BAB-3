# Exercise 3.15: Semi-supervised learning with three groups and one label each
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([[0, 1], [1, 1], [2, 0], [3, 1], [10, 5], [11, 6], [12, 4], [13, 5], [5, 10], [6, 11], [7, 10], [6, 9]])
labels = np.full(len(X), -1.0)
labels[0], labels[4], labels[8] = 0, 1, 2
model = LabelSpreading(kernel="knn", alpha=0.8).fit(X, labels)
print("Predicted labels:", model.transduction_)

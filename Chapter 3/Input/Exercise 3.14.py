# Exercise 3.14: Semi-supervised learning with two extra points per group
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([[0, 1], [1, 1], [2, 0], [3, 1], [1, 0], [2, 1], [10, 5], [11, 6], [12, 4], [13, 5], [11, 4], [12, 6]])
labels = np.full(len(X), -1.0)
labels[0] = 0
labels[-1] = 1
model = LabelSpreading(kernel="knn", alpha=0.8).fit(X, labels)
print("Predicted labels:", model.transduction_)

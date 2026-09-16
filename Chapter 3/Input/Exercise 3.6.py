# Exercise 3.6: LDA with 2,000 samples and 6 features
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=2000, n_features=6, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print("Prediction:", clf.predict([[0, 0, 0, 0, 0, 0]]))
print("Training accuracy:", clf.score(X, y))

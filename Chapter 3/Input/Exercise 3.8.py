# Exercise 3.8: Decision tree classification on wine data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

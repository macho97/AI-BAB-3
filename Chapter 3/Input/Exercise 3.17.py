# Exercise 3.17: Voting ensemble with K-nearest neighbors
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
estimators = [("lr", LogisticRegression(random_state=1)), ("rf", RandomForestClassifier(n_estimators=50, random_state=1)), ("gnb", GaussianNB()), ("svc", SVC()), ("knn", KNeighborsClassifier())]
ensemble = VotingClassifier(estimators=estimators, voting="hard")
scores = cross_val_score(ensemble, X, y, scoring="accuracy", cv=5)
print("Ensemble accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

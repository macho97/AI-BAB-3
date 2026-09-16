# Exercise 3.9: Random forest classification on discretized diabetes targets
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

X, y = load_diabetes(return_X_y=True)
y_class = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile").fit_transform(y.reshape(-1, 1)).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25, random_state=0, stratify=y_class)
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

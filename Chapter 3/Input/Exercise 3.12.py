# Exercise 3.12: Multiple linear regression on the Linnerud dataset
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

linnerud = load_linnerud()
X, y = linnerud.data, linnerud.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
model = LinearRegression().fit(X_train, y_train)
print("Feature names:", linnerud.feature_names)
print("Target names:", linnerud.target_names)
print("Predictions:\n", model.predict(X_test))
print("R-squared:", model.score(X_test, y_test))

# Exercise 3.18: California housing regression
# Uses scikit-learn because the book's auto_ml package is incompatible with Python 3.14.
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
print("R-squared:", model.score(X_test, y_test))

# Example 3.26 AutoML-style regression
# The book's auto_ml package is incompatible with Python 3.14.
# This current Scikit-Learn alternative automatically trains a robust regressor.
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("R-squared score:", model.score(X_test, y_test))

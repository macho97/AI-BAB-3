# Exercise 3.21: LazyPredict regression using diabetes data
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
models, predictions = LazyRegressor().fit(X_train, X_test, y_train, y_test)
print(models)

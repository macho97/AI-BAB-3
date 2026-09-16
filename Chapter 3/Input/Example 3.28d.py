# Example 3.28d LazyPredict regression
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# Use the built-in dataset so this program does not need to download data.
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Limit the comparison to fast regressors instead of running every available model.
reg = LazyRegressor(regressors=[LinearRegression, Ridge, RandomForestRegressor], random_state=1)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

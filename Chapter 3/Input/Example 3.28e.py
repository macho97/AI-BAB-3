# Example 3.28e LazyPredict regression plot
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
models, predictions = LazyRegressor(
    regressors=[LinearRegression, Ridge, RandomForestRegressor], random_state=1
).fit(X_train, X_test, y_train, y_test)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models["R-Squared"], "-s")
plt.xlabel("Model")
plt.ylabel("R-Squared")
plt.title("LazyPredict Regression Performance")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# Exercise 3.5: Save and load a Naive Bayes model
import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
model = GaussianNB()
model.fit(X, y)
joblib.dump(model, "naive_bayes_iris_model.joblib")

loaded_model = joblib.load("naive_bayes_iris_model.joblib")
print(loaded_model.predict([[5.0, 3.4, 1.5, 0.4]]))

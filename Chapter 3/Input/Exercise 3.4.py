# Exercise 3.4: Histograms of selected breast-cancer features
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
features = ["mean radius", "mean texture", "mean perimeter", "mean smoothness"]
df[features].hist(figsize=(10, 7), bins=20)
plt.suptitle("Breast Cancer Feature Histograms")
plt.tight_layout()
plt.show()

# Example 3.3 Python SVM Iris CSV Classifications
from pathlib import Path

import pandas as pd
from sklearn import datasets, svm

# Store the CSV next to this program, so it works regardless of the terminal folder.
csv_path = Path(__file__).with_name("iris.csv")
if not csv_path.exists():
    iris = datasets.load_iris(as_frame=True)
    df_iris = iris.data.copy()
    df_iris["species"] = [iris.target_names[target] for target in iris.target]
    df_iris.to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
X = df.values[:, :2]
s = df["species"]
d = {label: index for index, label in enumerate(sorted(set(s)))}
y = [d[label] for label in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict the flower for a given sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)

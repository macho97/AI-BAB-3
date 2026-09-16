# Exercise 3.3: Scatter plot of sepal length and sepal width
import matplotlib.pyplot as plt
import pandas as pd

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
df = pd.read_csv(url).dropna()
for species, group in df.groupby("species"):
    plt.scatter(group.iloc[:, 0], group.iloc[:, 1], label=species)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris: Sepal Length vs Sepal Width")
plt.legend()
plt.grid()
plt.show()

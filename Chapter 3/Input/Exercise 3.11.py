# Exercise 3.11: Linear regression with more data and complete plot labels
import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 4, 5, 6, 7, 7, 8, 10, 10, 12]
slope, intercept, *_ = stats.linregress(x, y)
line = [slope * value + intercept for value in x]
plt.scatter(x, y, label="Data points")
plt.plot(x, line, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()

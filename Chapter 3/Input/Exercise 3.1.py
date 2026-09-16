# Exercise 3.1: SVM with six samples
from sklearn import svm

X = [[170, 70, 10], [180, 80, 12], [175, 75, 9], [170, 65, 8], [160, 55, 7], [165, 60, 6]]
y = [0, 0, 0, 1, 1, 1]  # 0: Male, 1: Female
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[160, 60, 7]]))

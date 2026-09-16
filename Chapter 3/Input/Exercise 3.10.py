# Exercise 3.10: Compare classifiers on discretized diabetes targets
from sklearn.datasets import load_diabetes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_diabetes(return_X_y=True)
y = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile").fit_transform(y.reshape(-1, 1)).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
models = {"SVM": SVC(), "Naive Bayes": GaussianNB(), "LDA": LinearDiscriminantAnalysis(), "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1), "Decision Tree": DecisionTreeClassifier(random_state=0), "Random Forest": RandomForestClassifier(random_state=0), "KNN": KNeighborsClassifier(), "Neural Network": MLPClassifier(max_iter=1000, random_state=0)}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.3f}")

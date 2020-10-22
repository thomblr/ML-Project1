from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

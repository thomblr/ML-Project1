import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print("=== RESULTS ===")
print("Coefficients: ", lm.coef_)
print("Mean Squared Error : %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination : %.2f" % r2_score(y_test, y_pred))

plt.plot(X_test, y_pred, color='blue', linewidth=3)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 2

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 3

lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print("=== OLS ===")
print("Coefficients: ", lm.coef_)
print("Mean Squared Error : %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination : %.2f" % r2_score(y_test, y_pred))

# 4

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_lasso_pred = lasso.predict(X_test)
print(lasso.score(X_test, y_test))

print('=== Lasso Model ===')
print("Coeff: ", lasso.coef_)
print("MSE: %.2f" % mean_squared_error(y_test, y_lasso_pred))
print("R2: %.2f" % r2_score(y_test, y_lasso_pred))


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

# 2

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3

# Adding a column for the error
X_train = np.c_[X_train, np.ones(X_train.shape[0])]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]

xtrain = np.array(X_train) # X

xtrain_T = np.transpose(xtrain) # X^T

w = np.matmul(xtrain_T, xtrain) # X^T . X
inverted = np.linalg.inv(w) # (X^T . X)^-1
interm = np.dot(inverted, xtrain_T) # (X^T . X)^-1 . X^T
w_final = np.dot(interm, y_train) # (X^T . X)^-1 . X^T . y
print(w_final) # Print the best coefficients

y_final = np.dot(X_test, w_final)

score = r2_score(y_test, y_final)
print(score)

# 4

lasso = linear_model.Lasso(alpha=0.5)
lasso.fit(X_train, y_train)

y_lasso_pred = lasso.predict(X_test)

print('=== Lasso Model ===')
print("Coeff: ", lasso.coef_)
print("R2: %.4f" % r2_score(y_test, y_lasso_pred))

# 5

r2_array = []
best_r2 = 0
best_alpha = 0
for i in np.arange(0.01, 1, 0.01):
    lasso_bis = linear_model.Lasso(alpha=i)
    lasso_bis.fit(X_train, y_train)
    y_lasso_bis_pred = lasso_bis.predict(X_test)
    result_r2 = r2_score(y_test, y_lasso_bis_pred)

    if best_r2 < result_r2:
        best_r2 = result_r2
        best_alpha = i

    r2_array.append(result_r2)

plt.plot(np.arange(0.01, 1, 0.01), r2_array)
plt.xlabel("Value of Alpha in Lasse Model")
plt.ylabel("Value of the R2")
plt.title("Comparison of Alpha and R2 in a Lasso Model")
plt.show()

# 6

print("Best R2 : %.4f" % best_r2)
print("Best Alpha : %.4f" % best_alpha)

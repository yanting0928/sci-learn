from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy.lib.arraysetops as aso

lr = linear_model.LinearRegression()

rcv = linear_model.RidgeCV()

boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)
predicted2 = cross_val_predict(rcv, boston.data, y, cv=10)

print aso.setdiff1d(predicted, predicted2)

# which one is better?

print type(predicted2)
print rcv



fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



fig, ax = plt.subplots()
ax.scatter(y, predicted2)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured2')
ax.set_ylabel('Predicted2')
plt.show()
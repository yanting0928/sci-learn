print(__doc__)

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

# Fit IsotonicRegression and LinearRegression models
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

ir = IsotonicRegression()
ir.fit(X_train, y_train)
y_pre = ir.predict(X_test)
r2_score_ir  = r2_score(y_test,y_pre)

print r2_score_ir

lr = LinearRegression()
lr.fit(X_train[:, np.newaxis], y_train)  # x needs to be 2d for LinearRegression
y_pre = lr.predict(X_test[:, np.newaxis])
r2_score_lr  = r2_score(y_test,y_pre)

print r2_score_lr

la = Lasso()
la.fit(X_train[:, np.newaxis], y_train)
y_pre = la.predict(X_test[:, np.newaxis])
r2_score_la  = r2_score(y_test,y_pre)

print r2_score_la
#assert (lr.fit(x[:, np.newaxis], y) == la.fit(x[:, np.newaxis], y))


# plot result

segments = [[[i, y[i]], [i, y[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(n))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.plot(x, la.predict(x[:, np.newaxis]), 'p-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit', 'Lasso'), loc='lower right')
plt.title('Isotonic regression')
plt.show()

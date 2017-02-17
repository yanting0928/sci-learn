from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris,make_hastie_10_2
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

iris = load_iris()


param_grid = {"base_estimator__criterion": ["gini", "entropy"],
          "base_estimator__splitter":   ["best", "random"],
          "n_estimators": [1, 2]}

dtc = DecisionTreeClassifier()

ada = AdaBoostClassifier(base_estimator=dtc)

X, y = make_hastie_10_2(n_samples=12000, random_state=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search_ada = GridSearchCV(ada, param_grid=param_grid, cv=10)

grid_search_ada.fit(x_train, y_train)

y_test_predict = grid_search_ada.predict(x_test)

r2_score =r2_score(y_test, y_test_predict)

print r2_score


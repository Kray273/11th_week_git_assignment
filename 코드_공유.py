import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(dt_clf, param_grid=dt_params, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print(f"Best Hyper-parameter {dt_grid.best_params_}")
print(f"Best Score {dt_grid.best_score_}")

xgb_clf = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_params = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7, 9, 15],
    'n_estimators': [50, 100, 200, 300]
}
xgb_grid = GridSearchCV(xgb_clf, param_grid=xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"Best parameters: {xgb_grid.best_params_}")
print(f"Best accuracy: {xgb_grid.best_score_}")

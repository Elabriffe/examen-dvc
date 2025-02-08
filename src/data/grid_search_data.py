import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import os


# 1. GridSearch pour trouver les meilleurs paramètres
X_train= pd.read_csv("./data/processed_data/X_train_scaled.csv")
y_train= pd.read_csv("./data/processed_data/y_train.csv")

pipeline = Pipeline(steps=[
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [10, 50, 100, 200, 500],
    'regressor__max_depth': [5, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

y_train=np.ravel(y_train)

grid_search.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres
joblib.dump(grid_search.best_params_, "./models/best_params.pkl")
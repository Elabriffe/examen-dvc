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


# 1. Entraînement du modèle avec les meilleurs paramètres
X_train= pd.read_csv("./data/processed_data/X_train_scaled.csv")
y_train= pd.read_csv("./data/processed_data/y_train.csv")

best_param = joblib.load("./models/best_params.pkl")

cleaned_params = {key.replace('regressor__', ''): value for key, value in best_param.items()}

best_model=RandomForestRegressor(random_state=42,**cleaned_params)

y_train=np.ravel(y_train)

best_model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné
joblib.dump(best_model, "./models/trained_model.pkl")
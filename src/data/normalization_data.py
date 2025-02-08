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


# 1. Normalisation des données
X_train= pd.read_csv("./data/processed_data/X_train.csv")
y_train= pd.read_csv("./data/processed_data/y_train.csv")
X_test= pd.read_csv("./data/processed_data/X_test.csv")
y_test= pd.read_csv("./data/processed_data/y_test.csv")

cols_robust = ["ave_flot_level"]
cols_minmax = ["ave_flot_air_flow"]
cols_standard = [col for col in X_train.columns if col not in cols_minmax and col not in ["iron_feed", "starch_flow"] and col not in cols_robust]

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), cols_robust),
        ('standard', StandardScaler(), cols_standard),
        ('minmax', MinMaxScaler(), cols_minmax)
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Sauvegarde des données normalisées
pd.DataFrame(X_train_scaled, columns=cols_robust+cols_minmax+cols_standard).to_csv("./data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=cols_robust+cols_minmax+cols_standard).to_csv("./data/processed_data/X_test_scaled.csv", index=False)
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


# 1. Chargement du jeu de données
s3_path = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

data = pd.read_csv(s3_path, storage_options={"anon": True})
data["date"] = pd.to_datetime(data["date"])
data = data[data.groupby(data['date'].dt.floor('D'))['date'].transform('size') >= 12]
data = data.groupby(pd.Grouper(key='date', freq='D')).mean()
data = data.dropna().reset_index()
data = data.iloc[:, 1:]

# 2. Séparation des variables explicatives (X) et de la variable cible (y)
X = data.drop("silica_concentrate", axis=1)
y = data["silica_concentrate"]

# 3. Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des ensembles
X_train.to_csv("/home/ubuntu/exam_dvc/examen-dvc/data/processed_data/X_train.csv", index=False)
X_test.to_csv("/home/ubuntu/exam_dvc/examen-dvc/data/processed_data/X_test.csv", index=False)
y_train.to_csv("/home/ubuntu/exam_dvc/examen-dvc/data/processed_data/y_train.csv", index=False)
y_test.to_csv("/home/ubuntu/exam_dvc/examen-dvc/data/processed_data/y_test.csv", index=False)
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


# 1. Évaluation du modèle
X_test= pd.read_csv("./data/processed_data/X_test_scaled.csv")
y_test= pd.read_csv("./data/processed_data/y_test.csv")

best_model = joblib.load("./models/trained_model.pkl")
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des scores
evaluation_metrics = {"MSE": mse, "R2": r2}
with open("./metrics/scores.json", "w") as f:
    json.dump(evaluation_metrics, f)

# 8. Sauvegarde des prédictions
predictions = pd.DataFrame({"Actual": np.ravel(y_test), "Predicted": np.ravel(y_pred)})
predictions.to_csv("./models/predictions.csv", index=False)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from src.task_4.data_processing import prepare_claim_severity_data

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name} Results → RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return rmse, r2

def train_and_compare_models(X_train,y_train, X_test, y_test):

    results = {}
    models = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results['LinearRegression'] = evaluate_model("Linear Regression", lr, X_test, y_test)
    models['LinearRegression'] = lr

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['RandomForest'] = evaluate_model("Random Forest", rf, X_test, y_test)
    models['RandomForest'] = rf

    # 3. XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    results['XGBoost'] = evaluate_model("XGBoost", xgb, X_test, y_test)
    models['XGBoost'] = xgb

    # Identify best model by lowest RMSE
    best_model_name = min(results, key=lambda k: results[k][0])
    best_model = models[best_model_name]


    return best_model, best_model_name, results

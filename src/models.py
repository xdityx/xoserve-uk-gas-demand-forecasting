from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["demand_lag_1"]


def train_linear_regression(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_random_forest(X: pd.DataFrame, y: pd.Series):
    rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1)
    
    rf.fit(X,y)
    return rf


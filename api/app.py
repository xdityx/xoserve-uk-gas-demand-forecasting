from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_loader import load_demand
from src.models import compare_models, train_arima, train_sarima


app = FastAPI(title="UK Gas Demand Forecast API")

BASE_DIR = Path(__file__).resolve().parents[1]
DEMAND_PATH = BASE_DIR / "data" / "raw" / "uk_gas_demand_daily.csv"


class ForecastRequest(BaseModel):
    days: int = Field(gt=0, description="Number of days to forecast")
    model_type: Literal["arima", "sarima"]


def _load_demand_series():
    if not DEMAND_PATH.exists():
        raise FileNotFoundError(f"Demand data not found at {DEMAND_PATH}")

    demand_df = load_demand(str(DEMAND_PATH)).sort_values("date")
    if demand_df.empty:
        raise ValueError("Demand data is empty")

    return demand_df.set_index("date")["demand_gwh"]


def _split_series(y, test_size: int = 14):
    if len(y) <= test_size:
        raise ValueError("Not enough observations to create a train/test split")
    return y.iloc[:-test_size], y.iloc[-test_size:]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        y = _load_demand_series()
        if request.model_type == "arima":
            predictions = train_arima(y, steps=request.days)
        else:
            predictions = train_sarima(y, steps=request.days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"forecast": [float(value) for value in predictions.tolist()]}


@app.get("/compare")
def compare():
    try:
        y = _load_demand_series()
        y_train, y_test = _split_series(y)
        results = compare_models(y_train, y_test)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return results

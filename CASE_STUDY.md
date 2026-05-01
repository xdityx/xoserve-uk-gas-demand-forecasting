# UK Gas Demand Forecasting — Case Study

## The Problem
Energy demand forecasting is hard because supply and demand must stay aligned under real operational constraints. If demand is underforecast, operators may need emergency balancing actions or expensive reserve procurement. If it is overforecast, they can overcommit capacity, waste renewable output, and run the grid less efficiently than planned.

Demand is also driven by multiple overlapping factors: temperature, day type, seasonality, and structural change in how energy is consumed. Classical univariate models such as ARIMA remain useful baselines, but they can struggle when the regime shifts because of electrification, climate volatility, or changing heating behavior. In energy, good forecasting is usually less about picking a complex algorithm and more about combining domain features with disciplined time-series evaluation.

## Why This Matters
Forecasting error has direct financial consequences. Energy companies make long-range capital decisions worth billions, while grid operators and traders make daily decisions about reserve capacity, balancing, nominations, and scheduling. Even a modest improvement in forecast accuracy can reduce operating cost materially.

This challenge becomes more important as renewable penetration grows. Intermittent supply increases the value of accurate demand expectations, and the climate transition is changing demand shape. That makes energy forecasting a strong example of where domain expertise matters more than generic machine learning.

## What I Built
I built a production-oriented time-series forecasting system for daily UK National Transmission System gas demand using public data. The project is designed as a workflow rather than a notebook-only exercise: load data, engineer features, train models, evaluate them with time-aware validation, and serve outputs through an API.

The data layer uses daily NTS demand observations and UK weather data, with temperature as the primary exogenous driver in the current implementation. For gas demand, temperature is often the most important operational signal because it directly affects heating load. The pipeline is modular, so broader exogenous inputs such as holidays, wind, or other weather variables can be added later without changing the overall structure.

Feature engineering is where the project becomes domain-specific. The most important feature is Heating Degree Days (HDD), a standard energy metric derived from temperature relative to a base threshold. HDD is more useful than raw temperature because it better reflects how heating demand behaves in practice. I also added lag features and rolling averages to capture persistence and weekly rhythm in the demand series. Together, these features encode both weather sensitivity and system memory.

For modeling, I used ARIMA(1,1,1) as the classical baseline and SARIMA as the seasonal extension. SARIMA is useful here because weekly structure matters in operational energy demand, not just long-run seasonality. To compare models honestly, I added rolling window cross-validation rather than relying on a naive random split. The project also includes holdout evaluation and FastAPI endpoints for `/forecast` and `/compare`, returning JSON predictions and model metrics that can be consumed by dashboards or downstream applications.

## Key Technical Insights
The biggest lesson is that rolling window cross-validation is mandatory for time-series work. A random train/test split leaks future information and makes the results look better than they really are. Proper evaluation must mirror production: train on the past, predict the future, then roll forward.

The second lesson is that domain features beat pure autoregression on this problem. HDD is a stronger signal than raw temperature because it captures the mechanism that drives heating demand. Weekly patterns also matter alongside longer seasonal structure, which is why SARIMA adds value over a purely non-seasonal baseline.

Another important insight is that metric choice shapes business usefulness. RMSE is valuable because it penalizes large misses, but percentage error often communicates operational impact more clearly. Finally, time-series models drift. Retraining windows and evaluation periods have to respect seasonal context, or the forecast will look stable on paper while degrading in practice.

## Architecture Highlights
The project follows a modular pipeline: load, engineer, train, evaluate, and serve. It uses UK public-sector demand and weather data rather than synthetic examples. The validation logic is time-series aware, so there is no leakage between train and test windows. The serving layer is built with FastAPI and exposes forecast and comparison endpoints through JSON. The repository is also Dockerized and backed by automated tests for loading, features, metrics, and models. That makes the project reproducible, testable, and ready for further production hardening.

## Real-World Example
Imagine a grid operator planning for winter gas demand. A classical ARIMA workflow might train on historical demand and produce a 30-day forecast with 8-10% MAPE, which is typical for a weather-sensitive commodity. To stay safe, the operator may reserve more balancing capacity than necessary, which increases cost.

With this system, the operator can combine HDD-driven weather sensitivity with seasonal demand structure and compare ARIMA versus SARIMA under realistic rolling validation. If that improves accuracy by even 2-3 percentage points, the operational value can be meaningful. Better forecasts support tighter reserve decisions, more efficient scheduling, and better coordination around renewable variability.

## Why I Built This
I built this project because time-series forecasting is a different discipline from the cross-sectional machine learning that dominates most portfolios. In energy, weather, seasonality, and operational context often matter more than model novelty. This case study shows that I can work with real public data, translate energy-domain concepts like HDD into features, evaluate models correctly, and expose the outputs through a production-friendly API.

## Results & Proof
- Real UK public-sector demand and weather data
- HDD, lag, and rolling-demand feature engineering
- ARIMA baseline plus SARIMA comparison
- Rolling window cross-validation for leakage-aware evaluation
- FastAPI endpoints for `/forecast` and `/compare`
- Three automated test files for features, metrics, and models
- Dockerized, reproducible project structure

## What's Next
Next steps include richer exogenous inputs such as holidays, electricity prices, wind generation, and broader weather signals. I would also add prediction intervals so operators can manage uncertainty rather than only point forecasts. Beyond that, the natural progression is scheduled retraining, drift monitoring, and cloud deployment for a continuously updated forecasting service.

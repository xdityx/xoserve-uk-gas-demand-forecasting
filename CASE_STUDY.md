# UK Gas Demand Forecasting - Case Study

## The problem

UK gas-network operators need demand forecasts that respect gas-day timing, weather sensitivity, weekly structure, and publication delays. Forecast error affects balancing, nominations, reserve decisions, and operating cost. The main challenge is not simply fitting a model; it is recreating the information that would genuinely have been available at each forecast origin.

## What I built

This repository implements a complete daily forecasting workflow:

1. Retrieve finalized National Gas NTS D+6 demand and Met Office HadCET observations.
2. Align demand and weather by gas day.
3. Build Heating Degree Days, lag-1, lag-7, and previous-seven-day rolling features.
4. Train persistence, linear regression, RandomForest, ARIMA, and SARIMA models.
5. Evaluate every model with fixed-horizon recursive rolling-origin backtests.
6. Serve dated forecasts, intervals, model metadata, and data-freshness metadata through FastAPI.
7. Refresh source data and validation results through a scheduled workflow.

The project deliberately separates exploratory notebooks from authoritative evaluation. Reproducible evidence lives in `reports/oos_results.json`.

## Leakage correction

The original rolling feature included the current day's target. A chronological train/test split cannot repair leakage already present in a feature row.

The corrected feature is:

```python
demand_roll_7 = demand.shift(1).rolling(7).mean()
```

Multi-day weather-aware forecasts are also recursive. After predicting day one, that prediction - not the unknown actual demand - becomes part of the history used for day two. This makes the evaluation protocol consistent with deployment.

## Evaluation design

The backtest uses five expanding rolling-origin folds with a fixed 14-day horizon. Test windows run from 2026-04-29 through 2026-07-07. Persistence uses the final known training demand for the whole horizon. ARIMA and SARIMA forecast each holdout directly. Regression models receive realized holdout temperatures and recursively generated demand lags.

Using realized temperature isolates demand-model quality, but it excludes weather-forecast error. Weather-aware scores are therefore an optimistic upper bound on operational performance.

## Results

| Model | Mean MAE (GWh) | Mean RMSE (GWh) |
|---|---:|---:|
| Persistence | 146.62 | 172.12 |
| Linear regression + HDD | 192.68 | 227.14 |
| RandomForest + HDD | 155.64 | 183.01 |
| ARIMA | 148.13 | 173.57 |
| SARIMA | **128.78** | **157.88** |

SARIMA performs best in the tested period. The corrected results do not support the earlier conclusion that the in-sample RandomForest was the strongest forecaster. That difference is important: model fit on known observations and recursive future performance are separate questions.

The large fold variation for the weather-aware regressions also suggests regime sensitivity. HDD and short-memory demand features alone are insufficient for stable 14-day recursive forecasts.

## Serving design

The API supports `arima`, `sarima`, `linear`, and `random_forest`.

ARIMA and SARIMA produce model-based prediction intervals. Weather-aware models require one future mean-temperature value per horizon day and return empirical residual intervals. Every response includes:

- forecast dates and GWh values;
- lower and upper uncertainty bounds;
- model name and interval method;
- forecast origin and generation timestamp;
- demand-source age and freshness status.

The service rejects forecasts by default when finalized demand is more than 14 days old. A caller must explicitly set `allow_stale=true` to override that protection.

## Operational data

National Gas publication `PUBOB652` supplies finalized "Demand Actual, NTS, D+6" records. The updater requests the REST API in monthly chunks and upserts recent dates so revised publications can be captured.

Met Office HadCET is downloaded from its daily-updated authoritative text file. As of this case study, demand is current through 2026-07-07 and temperature through 2026-07-12, consistent with the respective publication schedules.

## Engineering proof

- 25 automated tests cover loading, feature leakage, metrics, models, recursive forecasting, backtesting, API metadata, freshness, and source-response parsing.
- Dependencies are pinned for reproducibility.
- Docker starts the FastAPI service and includes the source data and validation report.
- A scheduled GitHub workflow refreshes data, regenerates validation, runs tests, and commits changed artifacts.
- The API comparison endpoint returns the persisted report rather than fitting expensive models during a request.

## Limitations and next steps

The most valuable next improvements are demand-weighted regional weather forecasts, holiday and calendar effects, daily calorific-value conversion, formally calibrated intervals, drift monitoring, and evaluation with archived weather forecasts rather than realized temperature.

The central lesson is methodological: honest time-series performance depends more on information timing and recursive evaluation than on selecting a more complicated estimator.

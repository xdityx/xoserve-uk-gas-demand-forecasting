# UK Gas Demand Forecasting (NTS) – Weather-Aware Model

## Problem Statement
This project forecasts daily UK gas demand on the National Transmission System (NTS) using
historical demand data and weather information. Accurate short-term gas demand forecasting
is critical for system balancing, operational planning, and market stability.

The objective is to evaluate whether incorporating real UK weather data improves forecast
accuracy compared to a strong naïve baseline.

## Industry Context (UK Gas & Xoserve alignment)
UK gas operates on a gas-day basis, with demand published at multiple revision stages.
This project uses NTS Actual D+6 demand data, which represents near-finalized actuals
commonly relied upon for analytical validation.

Demand is modeled at the NTS level rather than LDZ-level to reflect system-wide forecasting
use cases relevant to network operators and central balancing.

## Data Sources
Gas Demand:
- Source: National Gas
- Dataset: NTS Actual Demand (D+6)
- Frequency: Daily
- Unit conversion: mscm → GWh

Weather:
- Source: UK Met Office (Central England Temperature – CET)
- Metric: Daily mean temperature (°C)
- Justification: CET is a widely used proxy for national-level UK temperature and is
  commonly applied in UK energy demand studies.

## Feature Engineering
The following features were engineered:

- Heating Degree Days (HDD), using a UK-standard base temperature of 15.5°C
- Demand lag features (t−1, t−7)
- Rolling averages (7-day)

HDD captures exogenous weather-driven demand, while lagged and rolling features
capture system inertia and weekly patterns.

## Baseline Model
A naïve persistence baseline was defined using demand from the previous day (lag-1).
This represents a strong and commonly used benchmark for short-term energy forecasting.

## Weather-Aware Model
A linear regression model was trained using HDD and demand history features.
This model was chosen to prioritise interpretability and alignment with operational
decision-making processes.

## Results
Model performance was evaluated using Mean Absolute Error (MAE):

- Baseline (lag-1): 140.47 GWh
- Linear Regression (with HDD): 129.74 GWh

Incorporating weather information resulted in a meaningful reduction in forecast error
relative to a strong baseline.

## Key Insights
- UK gas demand shows a strong negative correlation with temperature.
- Heating Degree Days are the dominant external driver of demand.
- Simple, interpretable models can deliver material performance gains over naïve methods.

## Limitations
- Temperature is represented using a national proxy rather than spatially weighted demand.
- No explicit calendar effects (e.g. holidays) were modeled.
- Models were evaluated in-sample and are intended for analytical demonstration.

## How This Could Be Extended
Future extensions could include:
- LDZ-level modeling
- Calendar and holiday effects
- Tree-based models for non-linear interactions
- Probabilistic forecasting and uncertainty bounds


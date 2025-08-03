📌 Project Overview
This project focuses on predicting the next day's average temperature deviation using historical meteorological data from multiple weather stations.
The goal is to predict how much warmer or cooler the next day will be compared to the climatological temperature (historical average).
We engineered advanced weather features and combined multiple regression models using a Stacking Regressor to maximize accuracy.

🎯 Objectives
Handle multi-year meteorological data with missing values and anomalies

Perform advanced feature engineering to capture daily, seasonal, and spatial weather patterns

Train base models (XGBoost, CatBoost) and combine them using Ridge Regression as meta-learner

Optimize hyperparameters using Optuna to minimize RMSE

Generalize model performance on unseen stations (Paju, Suwon)

🛠️ Techniques Used
Feature Engineering:

Temporal features: day of year, week of year, weekday/weekend

Cyclical encoding for seasonal periodicity (sin, cos transforms)

Night-time interaction features (temperature × humidity, vapor pressure changes)

Altitude and geographic coordinate combinations (lat × lon, lat ± lon)

Polynomial feature combinations

Data Processing:

Missing value handling:

-9999 → NaN

Zero-filling for certain variables (e.g., snow depth, sunshine duration)

Linear interpolation for continuous variables

Outlier clipping with optimized bounds

One-hot encoding for categorical variables

Models:

XGBoost

CatBoost (efficient categorical handling)

Ridge Regression (meta-learner for stacking)

Hyperparameter Optimization:

Optuna search for best model parameters and outlier thresholds

Evaluation Metrics:

RMSE (Root Mean Squared Error)

R² Score

📂 Dataset
Training Set:

Stations: Dongducheon, Seoul, Ganghwa, Incheon, Icheon, Yangpyeong

Time: 2019–2024

Test Set:

Stations: Paju, Suwon

Features:

19 weather variables (cloud cover, dew point, humidity, pressure, wind, etc.)

Climatology temperature (historical average for that date)

Target:

target = next_day_avg_temp - climatology_temp

🚧 Challenges & Improvements
Early models (single XGBoost or CatBoost) underperformed (R² ≈ 0.80)

Severe overfitting mitigated by:

Residual analysis

Ridge regularization in meta-learner

Feature clipping and careful missing value handling

Advanced temporal and interaction features significantly improved generalization

Stacking approach boosted performance to R² = 0.842, outperforming single models

Lessons learned:

Combining models with diverse strengths (CatBoost for categorical, XGBoost for complex splits) is effective

Advanced feature engineering (time, location, interactions) can rival model complexity in importance

📊 Results
XGBoost Alone: R² = 0.800

CatBoost Alone: R² ≈ 0.81

Stacking (XGBoost + CatBoost → Ridge):

R² = 0.84217

RMSE significantly reduced

Model demonstrated strong generalization to unseen stations (Paju, Suwon)

🔍 Additional Result Analysis
During final evaluation:

Stacking achieved a 4–5% improvement in R² over single models

Advanced feature sets (night-time interactions, altitude-adjusted temps) were top contributors

Hyperparameter tuning with Optuna refined outlier clipping, leading to better stability

Missing value imputation and temporal encoding reduced bias in cold and humid conditions

💡 Key Insight:
Feature engineering and stacking synergistically improved model accuracy and generalization.
Weather prediction benefits from combining multiple learners and rich temporal/geospatial features rather than relying solely on one complex model.

🚀 How to Run
Mount datasets (train/test/station info)

Run preprocessing.py → handles missing values, creates features

Run train_models.py → trains XGBoost, CatBoost, and Ridge Stacking

Run predict.py → generates submission CSV for test stations

🔑 Future Improvements
Incorporate lagged weather variables for time-series context

Experiment with neural networks (LSTM, Temporal CNN)

Integrate external data sources (satellite, climate indices)

Advanced imputation using KNN or multivariate methods

🔑 Author
Hyunbin Ki (Model design, feature engineering, code development)

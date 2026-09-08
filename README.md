# Dubai Real Estate Intelligence

An end-to-end machine learning project for predicting Dubai real-estate transaction values using historical property transaction data.

The project covers the complete ML workflow from data exploration and preprocessing to model training, evaluation, inference, automated testing, CI, and API serving.

---

## Project Overview

Dubai's real-estate market contains transactions across different property types, areas, procedures, ownership categories, projects, and transaction values.

The objective of this project is to build a machine learning system that can estimate the transaction value of a Dubai property based on available transaction and property characteristics.

The project is designed as an end-to-end production-oriented ML workflow rather than a standalone notebook experiment.

---

## Objectives

- Explore and understand Dubai real-estate transaction data.
- Clean and prepare the dataset for machine learning.
- Engineer useful numerical and categorical features.
- Compare multiple regression approaches.
- Handle the highly skewed transaction-value distribution.
- Evaluate model performance using multiple metrics.
- Analyze prediction errors across transaction-value segments.
- Build a reusable inference pipeline.
- Expose predictions through a FastAPI service.
- Add automated tests.
- Add GitHub Actions CI.
- Containerize the application with Docker.

---

## Dataset

The dataset contains Dubai real-estate transaction records.

The working dataset contains:

- 148,401 transactions
- 21 original columns

Important features include:

| Feature | Description |
|---|---|
| `GROUP_EN` | Transaction group |
| `PROCEDURE_EN` | Transaction procedure |
| `IS_OFFPLAN_EN` | Off-plan status |
| `IS_FREE_HOLD_EN` | Freehold status |
| `USAGE_EN` | Property usage |
| `AREA_EN` | Dubai area |
| `PROP_TYPE_EN` | Property type |
| `PROP_SB_TYPE_EN` | Property sub-type |
| `TRANS_VALUE` | Transaction value |
| `PROCEDURE_AREA` | Procedure area |
| `ACTUAL_AREA` | Actual property area |
| `ROOMS_EN` | Number/type of rooms |
| `PARKING` | Parking information |
| `NEAREST_METRO_EN` | Nearest metro station |
| `NEAREST_MALL_EN` | Nearest mall |
| `NEAREST_LANDMARK_EN` | Nearest landmark |
| `PROJECT_EN` | Project |
| `YEAR` | Transaction year |
| `MONTH` | Transaction month |
| `DAY_OF_WEEK` | Transaction day of week |

Large raw and processed datasets are intentionally excluded from Git.

---

## Project Structure

```text
dubai-real-estate-intelligence/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│
├── models/
│   ├── preprocessor.pkl
│   └── random_forest_log_model.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_model_inference.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Machine Learning Workflow

```text
Raw Transaction Data
        │
        ▼
Data Exploration
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train / Test Split
        │
        ▼
Categorical + Numerical Preprocessing
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Error Analysis
        │
        ▼
Saved Model + Preprocessor
        │
        ▼
Inference Pipeline
        │
        ▼
FastAPI
```

---

## Feature Engineering

The dataset contains both categorical and numerical variables.

### Categorical Features

- Transaction procedure
- Off-plan status
- Freehold status
- Usage
- Area
- Property type
- Property subtype
- Rooms
- Parking
- Nearest metro
- Nearest mall
- Nearest landmark
- Project

### Numerical Features

- Procedure area
- Actual area
- Year
- Month
- Day of week

Categorical features are transformed using the preprocessing pipeline, while numerical features are passed through the numerical preprocessing workflow.

---

## Target Transformation

The transaction-value distribution is highly right-skewed, with extremely large transactions.

The original target contains values reaching into the billions of AED.

To reduce the impact of this skew during model training, a logarithmic transformation was evaluated:

```python
y_log = np.log1p(y)
```

Predictions are converted back to AED using:

```python
prediction = np.expm1(prediction_log)
```

---

## Models Evaluated

Several models were evaluated during the modeling stage.

| Model | MAE (AED) | RMSE (AED) | R² |
|---|---:|---:|---:|
| Random Forest | 1,100,590.10 | 17,507,614.15 | 0.3325 |
| Random Forest + Log Target | 979,068.94 | 13,817,712.06 | 0.5842 |
| CatBoost + Log Target | 1,048,076.96 | 16,451,414.60 | 0.4106 |
| CatBoost + Log Target (Sales Only) | 805,576.46 | 16,051,546.49 | 0.2623 |

### Current Selected Model

The current inference pipeline uses:

**Random Forest + Log Target**

Overall test-set performance:

- **MAE:** AED 979,068.94
- **RMSE:** AED 13,817,712.06
- **R²:** 0.5842

The model was selected based on the overall evaluation results rather than relying on a single metric.

---

## Error Analysis

The transaction-value distribution contains significant high-value outliers.

The 99th percentile transaction-value threshold was:

```text
AED 31,500,000
```

Performance was also examined by price segment.

| Price Segment | Count | Median Absolute Error (AED) | Median Error % |
|---|---:|---:|---:|
| < 1M | 9,882 | 126,113.90 | 8.83% |
| 1M - 3M | 13,627 | 267,882.30 | 8.45% |
| 3M - 10M | 4,822 | 1,127,925.00 | 11.99% |
| 10M - 50M | 1,200 | 4,841,940.00 | 13.06% |
| 50M+ | 150 | 86,092,500.00 | 39.85% |

Overall:

```text
Median Absolute Percentage Error: 9.42%

Predictions within ±10%: 52.07%
Predictions within ±20%: 73.55%
Predictions within ±30%: 84.41%
```

The analysis shows that prediction accuracy varies substantially with transaction value, particularly for very high-value transactions.

---

## Feature Importance

The model identified several important features.

The strongest feature was:

```text
PROCEDURE_AREA
```

Other important features included:

- Transaction procedure
- Transaction group
- Actual area
- Freehold status
- Nearest mall
- Usage
- Month
- Area
- Nearest landmark
- Rooms

The importance analysis also highlighted the strong relationship between `PROCEDURE_AREA` and `ACTUAL_AREA`.

---

## Inference

The trained preprocessing pipeline and model are saved as model artifacts.

```text
models/
├── preprocessor.pkl
└── random_forest_log_model.pkl
```

The inference code is implemented in:

```text
src/predict.py
```

A sample property produced:

```text
Predicted Transaction Value: AED 930,940.96
```

---

## FastAPI

The model is exposed through a FastAPI application.

Application entry point:

```text
src/api.py
```

Run locally:

```powershell
python -m uvicorn src.api:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

FastAPI automatically provides interactive API documentation through Swagger UI.

---

## Example API Response

```json
{
  "predicted_transaction_value_aed": 930940.96
}
```

---

## Testing

The project uses `pytest` for automated testing.

Run the test suite:

```powershell
python -m pytest
```

Current test result:

```text
1 passed
```

The tests verify that a realistic property request can pass through the prediction pipeline and return a valid positive prediction.

The FastAPI endpoint is also tested using FastAPI's test client.

---

## Continuous Integration

GitHub Actions is configured through:

```text
.github/workflows/ci.yml
```

The CI pipeline:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs dependencies.
4. Runs the automated test suite.

```text
git push
    │
    ▼
GitHub Actions
    │
    ▼
Install dependencies
    │
    ▼
Run pytest
    │
    ├── PASS → CI succeeds
    │
    └── FAIL → CI fails
```

---

## Docker

The project includes a `Dockerfile` for containerizing the FastAPI application.

Docker support is included so the API can eventually be run consistently across development, testing, and deployment environments.

---

## Limitations

The current model has several limitations:

- Very high-value transactions are difficult to predict accurately.
- Extreme outliers have a substantial effect on RMSE.
- The dataset contains transaction types such as mortgages and gifts in addition to sales.
- Historical transaction data may not fully represent current market conditions.
- The current model is not a real-time property valuation system.
- External economic variables are not currently included.

---

## Future Improvements

Potential improvements include:

- Hyperparameter optimization.
- Better handling of extreme-value transactions.
- Separate models for different property types.
- Geographic feature engineering.
- Incorporating Dubai market and economic indicators.
- Time-based validation.
- Model monitoring.
- Prediction confidence intervals.
- Model versioning.
- Docker-based deployment.
- Cloud deployment.
- Production monitoring and logging.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- FastAPI
- Pytest
- Jupyter
- Git
- GitHub Actions
- Docker

---

## Status

**Development / Production-oriented prototype**

The project currently includes:

- Data exploration
- Machine learning modeling
- Model evaluation
- Error analysis
- Saved model artifacts
- Inference pipeline
- FastAPI prediction API
- Automated tests
- GitHub Actions CI
- Docker configuration

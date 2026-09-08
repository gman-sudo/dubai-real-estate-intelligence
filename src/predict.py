# src/predict.py

import numpy as np
import pandas as pd
import joblib


# Load the trained model and preprocessing pipeline.
model = joblib.load("models/random_forest_log_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


def predict_transaction_value(property_data):
    # Convert the input dictionary into a one-row DataFrame.
    property_df = pd.DataFrame([property_data])

    # Apply the same preprocessing used during model training.
    property_processed = preprocessor.transform(property_df)

    # Predict in log scale.
    prediction_log = model.predict(property_processed)

    # Convert the prediction back to AED.
    prediction_aed = np.expm1(prediction_log)[0]

    return prediction_aed
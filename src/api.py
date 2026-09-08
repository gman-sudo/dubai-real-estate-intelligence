from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_transaction_value


# Create the FastAPI application.
app = FastAPI(
    title="Dubai Real Estate Intelligence API",
    version="1.0.0"
)


# Define the structure of a property prediction request.
class PropertyInput(BaseModel):
    GROUP_EN: str
    PROCEDURE_EN: str
    IS_OFFPLAN_EN: str
    IS_FREE_HOLD_EN: str
    USAGE_EN: str
    AREA_EN: str
    PROP_TYPE_EN: str
    PROP_SB_TYPE_EN: str
    PROCEDURE_AREA: float
    ACTUAL_AREA: float
    ROOMS_EN: str
    PARKING: str
    NEAREST_METRO_EN: str
    NEAREST_MALL_EN: str
    NEAREST_LANDMARK_EN: str
    PROJECT_EN: str
    YEAR: int
    MONTH: int
    DAY_OF_WEEK: int


# Health-check endpoint.
@app.get("/")
def root():
    return {
        "message": "Dubai Real Estate Intelligence API is running"
    }


# Property price prediction endpoint.
@app.post("/predict")
def predict(property_input: PropertyInput):

    # Convert the validated request into a dictionary.
    property_data = property_input.model_dump()

    # Generate the predicted transaction value.
    prediction = predict_transaction_value(property_data)

    return {
        "predicted_transaction_value_aed": round(prediction, 2)
    }
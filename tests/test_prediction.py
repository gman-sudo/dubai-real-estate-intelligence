# Import FastAPI's test client.
from fastapi.testclient import TestClient

# Import our API application.
from src.api import app


# Create a test client without starting the actual server.
client = TestClient(app)


def test_prediction_endpoint():
    # Send a realistic property to the prediction endpoint.
    property_data = {
        "GROUP_EN": "Mortgage",
        "PROCEDURE_EN": "Portfolio Mortgage Registration Pre-Registration",
        "IS_OFFPLAN_EN": "Off-Plan",
        "IS_FREE_HOLD_EN": "Free Hold",
        "USAGE_EN": "Residential",
        "AREA_EN": "BUSINESS BAY",
        "PROP_TYPE_EN": "Unit",
        "PROP_SB_TYPE_EN": "Flat",
        "PROCEDURE_AREA": 46.17,
        "ACTUAL_AREA": 46.17,
        "ROOMS_EN": "Studio",
        "PARKING": "1",
        "NEAREST_METRO_EN": "Buj Khalifa Dubai Mall Metro Station",
        "NEAREST_MALL_EN": "Dubai Mall",
        "NEAREST_LANDMARK_EN": "Downtown Dubai",
        "PROJECT_EN": "THE CRESTMARK",
        "YEAR": 2026,
        "MONTH": 7,
        "DAY_OF_WEEK": 2
    }

    # Call the API endpoint.
    response = client.post(
        "/predict",
        json=property_data
    )

    # Verify the API request was successful.
    assert response.status_code == 200

    # Extract the JSON response.
    result = response.json()

    # Verify that the prediction is present.
    assert "predicted_transaction_value_aed" in result

    # Verify that the prediction is positive.
    assert result["predicted_transaction_value_aed"] > 0
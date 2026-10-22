from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"


def test_model_loads_successfully():
    model = joblib.load(MODEL_PATH)

    assert model is not None


import pandas as pd


def test_model_returns_valid_prediction():
    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }])

    prediction = model.predict(customer)

    assert len(prediction) == 1
    assert prediction[0] in ["Yes", "No"]



from app import app


def test_dashboard_loads_successfully():
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert b"Customer Churn" in response.data


def test_dashboard_rejects_missing_fields():
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.post("/", data={})

    assert response.status_code == 200
    assert b"Please fill in all required customer fields." in response.data



def test_prediction_probability_is_valid():
    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2
    }])

    probabilities = model.predict_proba(customer)
    churn_probability = probabilities[0, 1]

    assert 0 <= churn_probability <= 1

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold

def predict_churn(customer_data):
    model, threshold = load_model()

    churn_probability = model.predict_proba(customer_data)[0, 1]

    prediction = (
        "Yes" if churn_probability >= threshold
        else "No"
    )

    return prediction, churn_probability

if __name__ == "__main__":
    new_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0
    }])

    prediction, probability = predict_churn(new_customer)

    print("Predicted churn:", prediction)
    print(f"Churn probability: {probability:.2%}")
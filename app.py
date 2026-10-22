
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# --------------------------------------------------
# 1. Application paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.pkl"

app = Flask(__name__)


# --------------------------------------------------
# 2. Load the trained model and decision threshold
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))


# --------------------------------------------------
# 3. Dashboard statistics
#    Verified values from the original dataset
# --------------------------------------------------

total_customers = 7043
churned_customers = 1869
retained_customers = 5174
churn_rate = 26.54

contract_churn_data = {
    "Month-to-month": 42.71,
    "One year": 11.27,
    "Two year": 2.83,
}


# --------------------------------------------------
# 4. Expected model input features
# --------------------------------------------------

FEATURE_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERICAL_FIELDS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


# --------------------------------------------------
# 5. Dashboard and churn prediction route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = None
    churn_probability = None
    error = None

    if request.method == "POST":
        form_data = request.form.to_dict()

        # Validate that every expected feature was submitted.
        missing_fields = [
            field
            for field in FEATURE_FIELDS
            if not form_data.get(field, "").strip()
        ]

        if missing_fields:
            error = "Please fill in all required customer fields."

        else:
            try:
                # Convert numeric inputs to numbers.
                for field in NUMERICAL_FIELDS:
                    form_data[field] = float(form_data[field])

                # Validate SeniorCitizen.
                if form_data["SeniorCitizen"] not in (0, 1):
                    raise ValueError(
                        "SeniorCitizen must be 0 or 1."
                    )

                # Validate tenure.
                if not 0 <= form_data["tenure"] <= 72:
                    raise ValueError(
                        "Tenure must be between 0 and 72 months."
                    )

                if not form_data["tenure"].is_integer():
                    raise ValueError(
                        "Tenure must be a whole number of months."
                    )

                # Validate monthly charges.
                if not 0 <= form_data["MonthlyCharges"] <= 200:
                    raise ValueError(
                        "Monthly charges must be between 0 and 200."
                    )

                # Validate total charges.
                if form_data["TotalCharges"] < 0:
                    raise ValueError(
                        "Total charges cannot be negative."
                    )

                # Validate categorical inputs.
                for field, allowed_values in CATEGORICAL_VALUES.items():
                    if form_data[field] not in allowed_values:
                        raise ValueError(
                            f"Invalid value for {field}."
                        )

                # Prepare exactly the 19 features expected by the model.
                customer_df = pd.DataFrame([
                    {
                        field: form_data[field]
                        for field in FEATURE_FIELDS
                    }
                ])

                # Predict the probability of churn.
                churn_probability = float(
                    model.predict_proba(customer_df)[0, 1]
                )

                # Classify using the saved decision threshold.
                prediction = (
                    "Yes"
                    if churn_probability >= threshold
                    else "No"
                )

            except (ValueError, TypeError) as exc:
                error = str(exc)

    # Render the dashboard with predictions and statistics.
    return render_template(
        "dashboard.html",
        prediction=prediction,
        probability=(
            round(churn_probability * 100, 2)
            if churn_probability is not None
            else None
        ),
        error=error,
        total_customers=total_customers,
        churned_customers=churned_customers,
        retained_customers=retained_customers,
        churn_rate=churn_rate,
        contract_churn=contract_churn_data,
    )


# --------------------------------------------------
# 6. Run the Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

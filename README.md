# Customer Churn Prediction System

A machine learning web application that predicts whether a telecom customer is likely to leave a service. It combines a trained classification pipeline with a Flask dashboard for customer-level predictions and churn analytics.

## Project Overview

Customer churn is an important business problem for subscription-based services. Identifying customers who may leave can help businesses investigate retention opportunities and improve customer experience.

This project uses the IBM Telco Customer Churn dataset to build and evaluate machine learning models and serve predictions through a web interface.

## Features

- Customer churn prediction using a trained machine learning pipeline
- Interactive Flask dashboard
- Customer churn and retention KPIs
- Churn-rate comparison by contract type
- Input validation and error messages
- Automated tests using pytest
- Configurable classification threshold

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- Pytest
- HTML and CSS

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Clean the data and prepare the target variable.
3. Explore customer characteristics and churn patterns.
4. Split the data into training and testing sets.
5. Preprocess numerical and categorical features.
6. Train and compare classification models.
7. Tune the model and evaluate classification thresholds.
8. Save the trained pipeline and use it in the Flask application.

## Model Evaluation

The tuned Logistic Regression model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 80.48% |
| Precision | 65.52% |
| Recall | 55.88% |
| F1-score | 60.32% |
| ROC-AUC | 84.12% |

These results are from the model evaluated at the default 0.50 threshold. The application also uses a lower threshold of 0.31 to identify more potential churners, trading some precision for higher recall.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   └── churn_threshold.pkl
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   └── predict.html
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

*Adjust this tree to match the files actually present in your project.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local address displayed in the terminal in your browser.

## Run Automated Tests

```bash
python -m pytest tests/test_model.py -v
```

The tests check model loading, prediction output, dashboard availability, missing form validation, and probability validity.

## Dataset

The project uses the IBM Telco Customer Churn dataset. The dataset contains customer account information, services, contract details, charges, and churn labels.

Ensure you have the dataset in the `data/` directory before running the application.

## Limitations

- Predictions are estimates, not guarantees that a customer will leave.
- Model performance depends on the quality and representativeness of the dataset.
- Churn-rate differences across customer groups do not establish causation.
- The model should be evaluated on relevant real-world data before business deployment.

## Future Improvements

- Add richer model-performance visualizations.
- Improve automated tests for invalid numeric and categorical values.
- Add a model monitoring and retraining workflow.
- Deploy the application to a hosting platform.

## Author

**Tharshini**

B.Tech Artificial Intelligence and Data Science

GitHub: [Tharshini-User](https://github.com/Tharshini-User)


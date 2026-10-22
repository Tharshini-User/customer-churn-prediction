from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    load_and_clean_data,
    split_features_target,
    split_train_test,
    create_preprocessor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.pkl"

def build_model():
    preprocessor = create_preprocessor()

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            C=10,
            max_iter=1000
        ))
    ])

    return model

def train_model():
    df = load_and_clean_data(DATA_PATH)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_model()

    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation")
    print("-" * 30)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print(
        "Precision:",
        precision_score(y_test, y_pred, pos_label="Yes")
    )

    print(
        "Recall:",
        recall_score(y_test, y_pred, pos_label="Yes")
    )

    print(
        "F1-score:",
        f1_score(y_test, y_pred, pos_label="Yes")
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            (y_test == "Yes").astype(int),
            y_prob
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["No", "Yes"]))

def save_model(model, threshold=0.31):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

    print("Model saved to:", MODEL_PATH)
    print("Threshold saved to:", THRESHOLD_PATH)

if __name__ == "__main__":
    model, X_test, y_test = train_model()

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\nModel training completed successfully.")
    print("Test data shape:", X_test.shape)
    print("Test target shape:", y_test.shape)
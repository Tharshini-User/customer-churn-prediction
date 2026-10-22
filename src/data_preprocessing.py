import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
    "PaymentMethod"
]

def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    return df

def split_features_target(df):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    return X, y

def split_train_test(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

if __name__ == "__main__":
    file_path = r"C:\Users\THARSHINI\Downloads\customer-churn-prediction\data\Telco-Customer-Churn.csv"

    df = load_and_clean_data(file_path)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = create_preprocessor()

    print("Data shape:", df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
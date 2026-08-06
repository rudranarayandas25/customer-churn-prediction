import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("customerID", axis=1, errors="ignore")
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({"Yes": 1, "No": 0, "Male": 0, "Female": 1})

    multi_val_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    for col in multi_val_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded

def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_categorical(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler, df.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(
        "/workspace/data/telco_churn.csv"
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

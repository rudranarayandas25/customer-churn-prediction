import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]

gender = np.random.choice(["Male", "Female"], size=n, p=[0.48, 0.52])

senior_citizen = np.random.choice([0, 1], size=n, p=[0.84, 0.16])

partner = np.random.choice(["Yes", "No"], size=n, p=[0.52, 0.48])
dependents = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

tenure_months = np.random.randint(0, 73, size=n)

phone_service = np.random.choice(["Yes", "No"], size=n, p=[0.90, 0.10])

multiple_lines = np.where(
    phone_service == "No",
    "No phone service",
    np.random.choice(["Yes", "No"], size=n, p=[0.45, 0.55]),
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.44, 0.21]
)

online_security = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.29, 0.71]),
)
online_backup = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
)
device_protection = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.31, 0.69]),
)
tech_support = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.25, 0.75]),
)
streaming_tv = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.44, 0.56]),
)
streaming_movies = np.where(
    internet_service == "No",
    "No internet service",
    np.random.choice(["Yes", "No"], size=n, p=[0.43, 0.57]),
)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
)

paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40])

payment_method = np.random.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    size=n,
    p=[0.36, 0.23, 0.22, 0.19],
)

monthly_charges = np.round(
    20 + 10 * (tenure_months / 72) + np.random.normal(50, 25, n), 2
)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

total_charges = np.where(
    tenure_months == 0,
    0.0,
    np.round(monthly_charges * tenure_months + np.random.normal(0, 50, n), 2),
)
total_charges = np.clip(total_charges, 0, None)

churn_prob = (
    -1.5
    - 1.0 * (tenure_months / 72)
    + 1.0 * (contract == "Month-to-month")
    + 0.7 * (payment_method == "Electronic check")
    + 0.5 * (internet_service == "Fiber optic")
    - 0.5 * (tech_support == "Yes")
    - 0.4 * (online_security == "Yes")
    + 0.25 * (paperless_billing == "Yes")
    + 0.02 * (monthly_charges - 60)
    + 0.3 * (senior_citizen == 1)
)

deterministic_churn = (churn_prob >= 0.0).astype(int)
probabilistic_churn = (np.random.random(n) < churn_prob).astype(int)
churn = np.where(np.random.random(n) < 0.7, deterministic_churn, probabilistic_churn)

df = pd.DataFrame(
    {
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure_months,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    }
)

df.to_csv("data/telco_churn.csv", index=False)

print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].mean():.2%}")
print(df.head())

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import joblib
import os


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {X_train.shape[0]} -> {X_resampled.shape[0]} samples")
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    neg_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'=' * 50}")
    print(f"  {model_name} Evaluation")
    print(f"{'=' * 50}")
    for k, v in results.items():
        if k != "model":
            print(f"  {k.capitalize():15s}: {v:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

    return results, y_proba


def compare_models(results_list):
    df = pd.DataFrame(results_list)
    df = df.set_index("model")
    print(f"\n{'=' * 60}")
    print("  Model Comparison Summary")
    print(f"{'=' * 60}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def save_model(model, scaler, model_name: str, base_dir: str = "/workspace/models"):
    os.makedirs(base_dir, exist_ok=True)
    joblib.dump(model, os.path.join(base_dir, f"{model_name}.joblib"))
    joblib.dump(scaler, os.path.join(base_dir, "scaler.joblib"))
    print(f"Model saved to {base_dir}/{model_name}.joblib")


if __name__ == "__main__":
    from preprocessing import prepare_data

    X_train, X_test, y_train, y_test, scaler, _ = prepare_data(
        "/workspace/data/telco_churn.csv"
    )

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    models = {
        "LogisticRegression": train_logistic_regression(X_train_res, y_train_res),
        "RandomForest": train_random_forest(X_train_res, y_train_res),
        "XGBoost": train_xgboost(X_train_res, y_train_res),
    }

    all_results = []
    best_model_name = None
    best_f1 = 0
    best_model = None
    best_proba = None

    for name, model in models.items():
        results, y_proba = evaluate_model(model, X_test, y_test, name)
        all_results.append(results)

        if results["f1_score"] > best_f1:
            best_f1 = results["f1_score"]
            best_model_name = name
            best_model = model
            best_proba = y_proba

    compare_models(all_results)

    save_model(best_model, scaler, best_model_name)
    print(f"\nBest model: {best_model_name} (F1={best_f1:.4f})")

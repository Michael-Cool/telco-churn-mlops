# src/train_and_serialize.py
# Reproducible end-to-end training + serialization with MLflow tracking & registry
# Final model: XGBoost; RandomForest logged as benchmark.

import json
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier

# ---- Konfiguration
RANDOM_STATE = 42
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-xgboost"

# ---- Projektpfade
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "encoded"  # train_encoded.csv / val_encoded.csv / test_encoded.csv
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PKL = MODELS_DIR / "final_model.pkl"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
FEATURES_JSON = MODELS_DIR / "model_input_features.json"

# ---- 🧩 Lokales MLflow Tracking
mlflow.set_tracking_uri("file://" + str(ROOT / "mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)


# ---- Helper-Funktionen
def load_splits():
    train = pd.read_csv(DATA_DIR / "train_encoded.csv")
    val = pd.read_csv(DATA_DIR / "val_encoded.csv")
    test = pd.read_csv(DATA_DIR / "test_encoded.csv")

    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"].astype(int)
    X_val, y_val = val.drop(columns=["Churn"]), val["Churn"].astype(int)
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"].astype(int)

    X_tr_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_tr_cv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_tr_cv, y_tr_cv


def scale_fit_transform(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return scaler, X_train_s, X_val_s, X_test_s


def smote_balance(X, y):
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


def eval_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def log_metrics_table(prefix, metrics):
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}_{k}", v)


def main():
    # --- Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, X_tr_cv, y_tr_cv = load_splits()
    feature_names = X_train.columns.tolist()

    # --- Scale
    scaler, X_train_s, X_val_s, X_test_s = scale_fit_transform(X_train, X_val, X_test)

    # --- Balance (SMOTE)
    X_train_bal, y_train_bal = smote_balance(X_train_s, y_train)

    # --- Parent run
    with mlflow.start_run(run_name="rf_benchmark_and_xgb_final", nested=False) as parent_run:
        mlflow.set_tags({
            "stage": "training",
            "random_state": RANDOM_STATE,
            "data_version": "encoded_splits",
        })

        # --- Random Forest benchmark
        with mlflow.start_run(run_name="rf_benchmark", nested=True):
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train_bal, y_train_bal)
            rf_proba = rf.predict_proba(X_test_s)[:, 1]
            rf_pred = (rf_proba >= 0.5).astype(int)
            rf_metrics = eval_metrics(y_test, rf_pred, rf_proba)

            mlflow.log_params({
                "model": "RandomForestClassifier",
                "n_estimators": rf.n_estimators,
                "max_depth": rf.max_depth,
                "min_samples_split": rf.min_samples_split,
            })
            log_metrics_table("rf", rf_metrics)
            mlflow.sklearn.log_model(rf, "rf_model")

        # --- XGBoost final (GridSearch + Registry)
        with mlflow.start_run(run_name="xgb_final", nested=True):
            xgb = XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                tree_method="hist",
            )

            param_grid = {
                "n_estimators": [300, 500, 800],
                "max_depth": [4, 5, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

            grid = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train_bal, y_train_bal)

            best_xgb = grid.best_estimator_
            mlflow.log_params({"search": "GridSearchCV", "cv": 5})
            mlflow.log_params(grid.best_params_)

            xgb_proba = best_xgb.predict_proba(X_test_s)[:, 1]
            xgb_pred = (xgb_proba >= 0.5).astype(int)
            xgb_metrics = eval_metrics(y_test, xgb_pred, xgb_proba)
            log_metrics_table("xgb", xgb_metrics)

            FEATURES_JSON.write_text(json.dumps(feature_names, indent=2))
            mlflow.log_artifact(FEATURES_JSON.as_posix())

            with open(SCALER_PKL, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(SCALER_PKL.as_posix())

            model_info = mlflow.xgboost.log_model(
                xgb_model=best_xgb,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            with open(FINAL_MODEL_PKL, "wb") as f:
                pickle.dump(best_xgb, f)

            print("✅ Saved local:", FINAL_MODEL_PKL)
            print("✅ Logged to MLflow:", model_info.model_uri)

        mlflow.set_tags({"final_model": "XGBoost", "benchmark": "RandomForest"})

    FEATURES_JSON.write_text(json.dumps(feature_names, indent=2))
    print("✅ Features saved:", FEATURES_JSON)


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()

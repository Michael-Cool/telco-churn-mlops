import time
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_data(path="data/processed"):
    # Trainingsdaten laden
    X_train = pd.read_csv(f"{path}/train.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv").values.ravel()
    return X_train, y_train


def train_and_log_model(model, model_name, X_train, y_train):
    # Neues MLflow-Experiment für Modelltraining
    mlflow.set_experiment("telco-model-training")

    with mlflow.start_run(run_name=model_name):

        # Modellparameter loggen
        mlflow.log_params(model.get_params())

        # Trainingsdauer messen
        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start
        mlflow.log_metric("training_time", training_time)

        # Modell speichern
        joblib.dump(model, f"models/{model_name}.pkl")
        mlflow.log_artifact(f"models/{model_name}.pkl")

        # Modell zusätzlich im MLflow-Format ablegen
        mlflow.sklearn.log_model(model, model_name)

        return model


if __name__ == "__main__":
    X_train, y_train = load_data()

    # Logistic Regression trainieren
    lr = LogisticRegression(max_iter=200)
    train_and_log_model(lr, "logistic_regression", X_train, y_train)

    # Random Forest trainieren
    rf = RandomForestClassifier(n_estimators=200)
    train_and_log_model(rf, "random_forest", X_train, y_train)

    # XGBoost trainieren
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss"
    )
    train_and_log_model(xgb, "xgboost", X_train, y_train)
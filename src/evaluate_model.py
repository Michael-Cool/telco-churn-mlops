import time
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt


def load_test_data(path="data/processed"):
    # Testdaten laden
    X_test = pd.read_csv(f"{path}/test.csv")
    y_test = pd.read_csv(f"{path}/y_test.csv")
    return X_test, y_test.values.ravel()


def plot_and_log_curve(y_true, y_proba, model_name, curve_type):
    # MLflow-Metrik zur Zielverteilung loggen
    mlflow.log_metric("y_true_mean", y_true.mean())

    # Kurven berechnen & visualisieren
    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr)
        plt.title(f"{model_name} ROC Curve")
        file = f"roc_{model_name}.png"
    else:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(rec, prec)
        plt.title(f"{model_name} PR Curve")
        file = f"pr_{model_name}.png"

    # Grafik speichern und in MLflow loggen
    plt.savefig(file)
    plt.close()
    mlflow.log_artifact(file)


def evaluate(model_name):
    # Neues MLflow-Experiment für Evaluation
    mlflow.set_experiment("telco-model-evaluation")

    with mlflow.start_run(run_name=model_name):

        # Modell laden
        model = joblib.load(f"models/{model_name}.pkl")
        X_test, y_test = load_test_data()

        # Vorhersagezeit messen
        start = time.time()
        y_proba = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - start
        mlflow.log_metric("prediction_time", pred_time)

        # Klassenvorhersagen erzeugen
        y_pred = (y_proba > 0.5).astype(int)

        # Metriken loggen
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("log_loss", log_loss(y_test, y_proba))

        # ROC- und PR-Kurven speichern
        plot_and_log_curve(y_test, y_proba, model_name, "roc")
        plot_and_log_curve(y_test, y_proba, model_name, "pr")


if __name__ == "__main__":
    # Alle Modelle evaluieren
    evaluate("logistic_regression")
    evaluate("random_forest")
    evaluate("xgboost")

import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    matthews_corrcoef,
)

# Ausgabeordner
os.makedirs("figures", exist_ok=True)

# Eine gemeinsame PDF-Datei für alle Plots
pdf_path = "figures/model_comparison_plots.pdf"
pdf = PdfPages(pdf_path)

# MLflow Client
client = MlflowClient()

experiment = client.get_experiment_by_name("telco-model-evaluation")
experiment_id = experiment.experiment_id

runs = client.search_runs(experiment_ids=[experiment_id])

# Nur die finalen Modelle
valid_models = ["random_forest", "xgboost"]
runs = [r for r in runs if r.info.run_name in valid_models]

model_names = []
metric_store = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": [],
    "mcc": [],
    "log_loss": [],
    "prediction_time": [],
}

# Testdaten für zusätzliche Metriken
X_test = pd.read_csv("data/processed/test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Modelle laden
models = {}

for run in runs:
    name = run.info.run_name
    model_names.append(name)

    # Modell laden
    model = joblib.load(f"models/{name}.pkl")
    models[name] = model

    # Vorhersagen berechnen
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # MLflow-Metriken
    metric_store["accuracy"].append(run.data.metrics.get("accuracy", np.nan))
    metric_store["f1"].append(run.data.metrics.get("f1", np.nan))
    metric_store["roc_auc"].append(run.data.metrics.get("auc", np.nan))
    metric_store["log_loss"].append(run.data.metrics.get("log_loss", np.nan))
    metric_store["prediction_time"].append(run.data.metrics.get("prediction_time", np.nan))

    # Berechnete Metriken
    metric_store["precision"].append(precision_score(y_test, y_pred))
    metric_store["recall"].append(recall_score(y_test, y_pred))
    metric_store["mcc"].append(matthews_corrcoef(y_test, y_pred))


def add_to_pdf():
    pdf.savefig()
    plt.close()


# Balkendiagramm-Helferfunktion
def plot_bar(metric_key, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, metric_store[metric_key], color="steelblue")
    plt.title(title)
    plt.xlabel("Modelle")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    add_to_pdf()


# ------------------------------------------------------------
# 1) GENAU in der Reihenfolge, die du wolltest
# ------------------------------------------------------------

plot_bar("accuracy", "Vergleich der Modelle anhand der Accuracy", "Accuracy")

plot_bar("precision", "Vergleich der Modelle anhand der Precision",
         "Precision (Genauigkeit der positiven Vorhersagen)")

plot_bar("recall", "Vergleich der Modelle anhand des Recalls",
         "Recall (Erkennungsrate der Churn-Kunden)")

plot_bar("f1", "Vergleich der Modelle anhand des F1-Scores", "F1-Score")

plot_bar("roc_auc", "Vergleich der Modelle anhand der ROC AUC",
         "Area Under the ROC Curve (AUC)")

plot_bar("mcc", "Vergleich der Modelle anhand des Matthews Correlation Coefficient (MCC)",
         "Matthews Correlation Coefficient")

plot_bar("log_loss", "Vergleich der Modelle anhand des Logarithmic Loss",
         "Logarithmic Loss")

plot_bar("prediction_time", "Vergleich der Modelle anhand der Predictionzeit",
         "Predictionzeit (Sekunden)")


# ------------------------------------------------------------
# PDF speichern
# ------------------------------------------------------------

pdf.close()

print(f"Alle Grafiken wurden erfolgreich in '{pdf_path}' gespeichert.")

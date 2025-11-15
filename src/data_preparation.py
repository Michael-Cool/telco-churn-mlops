import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


def prepare_data(raw_path: str, output_dir: str):
    # Neues MLflow-Experiment für Datenvorbereitung starten
    mlflow.set_experiment("telco-data-preparation")

    with mlflow.start_run():

        # Rohdaten einlesen
        df = pd.read_csv(raw_path)

        # TotalCharges ist als String gespeichert → in numerischen Typ umwandeln
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fehlende Werte mit Median ersetzen (robust bei Ausreißern)
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        # Basisinformationen loggen
        mlflow.log_metric("rows_raw", df.shape[0])
        mlflow.log_metric("cols_raw", df.shape[1])

        # Zielvariable binarisieren
        y = df["Churn"].map({"Yes": 1, "No": 0})
        # Feature-Matrix erstellen
        X = df.drop("Churn", axis=1)

        # Manuelle Definition numerischer und kategorialer Features
        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_features = [
            "gender","SeniorCitizen","Partner","Dependents","PhoneService",
            "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
            "Contract","PaperlessBilling","PaymentMethod"
        ]

        # Skalierung für numerische Daten
        numeric_transformer = StandardScaler()
        # One-Hot-Encoding für kategoriale Daten
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Preprocessing-Schritte kombinieren
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        # Pipeline definieren
        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        # Daten transformieren
        X_processed = pipeline.fit_transform(X)

        # Train/Val/Test-Split mit Stratifizierung
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Ausgabeordner anlegen
        os.makedirs(output_dir, exist_ok=True)
        # Preprocessor speichern (für spätere API-Predictions)
        joblib.dump(pipeline, f"{output_dir}/preprocessor.pkl")

        # Datensplits speichern
        pd.DataFrame(X_train).to_csv(f"{output_dir}/train.csv", index=False)
        pd.DataFrame(X_val).to_csv(f"{output_dir}/val.csv", index=False)
        pd.DataFrame(X_test).to_csv(f"{output_dir}/test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        # MLflow-Parameter loggen
        mlflow.log_param("numeric_features", numeric_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.log_param("target", "Churn (Yes=1, No=0)")
        mlflow.log_param("split_seed", 42)
        # Artefakte (Datensplits) speichern
        mlflow.log_artifact(output_dir)


if __name__ == "__main__":
    prepare_data("data/raw/IBMTelco_Datensatz.csv", "data/processed")
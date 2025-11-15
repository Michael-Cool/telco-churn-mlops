import joblib
import pandas as pd


def predict(input_dict, model_path="models/xgboost.pkl", preprocessor_path="data/processed/preprocessor.pkl"):
    # Modell und Preprocessor laden
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Eingabedaten in DataFrame umwandeln
    X = pd.DataFrame([input_dict])

    # Gleiche Transformation anwenden wie beim Training
    X_processed = preprocessor.transform(X)

    # Wahrscheinlichkeit und finale Vorhersage berechnen
    proba = model.predict_proba(X_processed)[0][1]
    pred = int(proba > 0.5)

    return {"prediction": pred, "probability": float(proba)}


if __name__ == "__main__":
    example = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 62,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.15,
        "TotalCharges": 3753.2
    }
    print(predict(example))

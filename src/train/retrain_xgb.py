import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import shutil

# === 1. Daten laden ===
train = pd.read_csv("data/cleaned/train_clean.csv")
val = pd.read_csv("data/cleaned/val_clean.csv")
test = pd.read_csv("data/cleaned/test_clean.csv")

df = pd.concat([train, val, test], ignore_index=True)
print("✅ Daten geladen:", df.shape)

# === 2. Zielvariable automatisch erkennen ===
target_candidates = [c for c in df.columns if "churn" in c.lower()]
if not target_candidates:
    raise ValueError("❌ Keine Zielspalte gefunden (Churn).")
target_col = target_candidates[0]
print(f"🔍 Gefundene Zielspalte: {target_col}")

# === 3. Zielvariable bereinigen ===
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
    .replace({"Yes": 1, "No": 0, "": np.nan, "nan": np.nan})
)
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
print("✅ Nach Reinigung:", df[target_col].value_counts().to_dict())

# Einheitliche Zielspalte für das Training
df["Churn"] = df[target_col]

# === 4. One-Hot-Encoding (kompatibel zur API) ===
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# === 5. Feature-Namen speichern ===
feature_names = [c for c in df_encoded.columns if c != "Churn"]
joblib.dump(feature_names, "models/feature_names.pkl")
print(f"💾 {len(feature_names)} Feature-Namen gespeichert unter models/feature_names.pkl")

# === 6. Train/Test-Split ===
X = df_encoded[feature_names]
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 7. Class Weight berechnen ===
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"⚖️  scale_pos_weight = {scale_pos_weight:.2f}")

# === 8. Modell trainieren ===
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)
model.fit(X_train, y_train)
print("✅ Training abgeschlossen.")

# === 9. Bewertung ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Modellbewertung:")
print(classification_report(y_test, y_pred))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))

# === 10. Backup der vorherigen Modellversion ===
model_dir = "models"
current_model = os.path.join(model_dir, "xgboost_model.pkl")
previous_model = os.path.join(model_dir, "previous_model.pkl")

if os.path.exists(current_model):
    shutil.copy(current_model, previous_model)
    print("🧩 Alte Modellversion gesichert als previous_model.pkl")

# === 11. Neues Modell speichern ===
joblib.dump(model, current_model)
print("✅ Neues Modell gespeichert unter models/xgboost_model.pkl")

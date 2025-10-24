import joblib
import pandas as pd
import xgboost as xgb

# Modell und Feature-Namen laden
model = joblib.load("models/xgboost_model.pkl")
features = joblib.load("models/feature_names.pkl")

# Feature Importances auslesen
importances = model.feature_importances_
df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
df_imp = df_imp.sort_values(by="Importance", ascending=False)

# Top 15 anzeigen
print("\n🏆 Top 15 Feature Importances:")
print(df_imp.head(15).to_string(index=False))

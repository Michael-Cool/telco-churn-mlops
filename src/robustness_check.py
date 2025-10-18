import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data", "processed")

train = pd.read_csv(os.path.join(data_path, "train.csv"))
test = pd.read_csv(os.path.join(data_path, "test.csv"))
val = pd.read_csv(os.path.join(data_path, "val.csv"))

df = pd.concat([train, test, val], axis=0)

X = df.drop("Churn", axis=1)
y = df["Churn"]

results = []
for seed in [1, 7, 21, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = XGBClassifier(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    results.append({
        "seed": seed,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "logloss": log_loss(y_test, y_proba)
    })

df_results = pd.DataFrame(results)
print(df_results)
print(df_results.describe())
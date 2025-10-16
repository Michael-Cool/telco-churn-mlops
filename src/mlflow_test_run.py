import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🧠 Tracking-URI (lokal)
mlflow.set_tracking_uri("file:///Users/michaelnatterer/telco-churn-eda/mlruns")
mlflow.set_experiment("mlflow_diagnostic_test")

# 🧪 Dummy-Daten
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 🌲 Einfaches Modell
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 🚀 MLflow Run starten
with mlflow.start_run(run_name="diagnostic_rf_test"):
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Test Run completed. Accuracy: {acc:.3f}")
    print("Run logged successfully to MLflow.")

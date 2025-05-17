import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import joblib

mlflow.set_experiment("AQI_Classification_CI")

# Load dataset
df = pd.read_csv("aqi_preprocessing.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Category'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

input_example = X_train.iloc[:5]

n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

best_accuracy = 0
best_model = None

with mlflow.start_run():
    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            # Buat nama parameter gabungan untuk membedakan log
            param_label = f"{n_estimators}_{max_depth}"

            # Log parameter sebagai dictionary agar tetap clean
            mlflow.log_param(f"n_estimators_{param_label}", n_estimators)
            mlflow.log_param(f"max_depth_{param_label}", max_depth)

            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            mlflow.log_metric(f"accuracy_{param_label}", accuracy)
            mlflow.log_metric(f"precision_{param_label}", precision)
            mlflow.log_metric(f"recall_{param_label}", recall)
            mlflow.log_metric(f"f1_score_{param_label}", f1)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name="RandomForestAQI"
                )

# Simpan model terbaik lokal
if best_model:
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.joblib")

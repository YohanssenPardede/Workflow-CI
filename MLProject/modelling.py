import mlflow
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("AQI_Classification_CI")

df = pd.read_csv("aqi_preprocessing.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Category'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Log some dataset related metrics
mlflow.log_metric("train_samples", len(X_train))
mlflow.log_metric("test_samples", len(X_test))
mlflow.log_metric("num_features", len(features))

n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

best_accuracy = 0
best_model = None
iteration = 0

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        iteration += 1
        param_label = f"{n_estimators}_{max_depth}"
        
        mlflow.log_param(f"n_estimators_{param_label}", n_estimators)
        mlflow.log_param(f"max_depth_{param_label}", max_depth)
        
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        training_duration = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric(f"accuracy_{param_label}", accuracy)
        mlflow.log_metric(f"training_duration_{param_label}", training_duration)
        mlflow.log_metric(f"iteration", iteration)
        mlflow.log_metric(f"is_best_model_{param_label}", 1 if accuracy > best_accuracy else 0)
        mlflow.log_metric(f"accuracy_diff_{param_label}", accuracy - best_accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="RandomForestAQI")

if best_model:
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, os.path.join("models", "best_model.joblib"))

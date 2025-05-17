import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

mlflow.set_experiment("AQI_Classification_CI")

df = pd.read_csv("aqi_preprocessing.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Category'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

best_accuracy = 0
best_model = None

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        param_label = f"{n_estimators}_{max_depth}"
        
        with mlflow.start_run():
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="RandomForestAQI")

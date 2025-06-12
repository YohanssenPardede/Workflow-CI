import mlflow
import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- MLflow Setup ---
mlflow.set_experiment("AQI_Classification_CI")

# --- Data Loading and Preparation ---
df = pd.read_csv("aqi_preprocessing.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Category'

X = df[features]
y = df[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Jumlah kategori AQI unik: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- MLflow Logging: Dataset Metrics ---
# These can be logged outside the inner run as they are part of the overall process
mlflow.log_metric("train_samples", len(X_train))
mlflow.log_metric("test_samples", len(X_test))
mlflow.log_metric("num_features", len(features))
mlflow.log_metric("num_target_classes", num_classes)

# --- TensorFlow/Keras Model Definition and Training ---
# Change this line:
with mlflow.start_run(nested=True): # <--- ADDED nested=True HERE
    # Log Keras model parameters
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss_function", "sparse_categorical_crossentropy")
    mlflow.log_param("dense_layer_1_units", 64)
    mlflow.log_param("dropout_rate_1", 0.2)
    mlflow.log_param("dense_layer_2_units", 32)
    mlflow.log_param("dropout_rate_2", 0.2)
    mlflow.log_param("output_layer_units", num_classes)
    mlflow.log_param("activation_output", "softmax")

    epochs = 50
    batch_size = 64
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    start_time = time.time()
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        verbose=1)
    training_duration = time.time() - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    mlflow.log_metric("final_test_accuracy", accuracy)
    mlflow.log_metric("final_test_loss", loss)

    mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name="TensorFlowAQI")
    print("Model TensorFlow dicatat ke MLflow.")

    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "air_quality_dnn_model.h5")
    model.save(model_save_path)
    print(f"Model TensorFlow disimpan secara lokal di: {model_save_path}")

print("\nEksperimen selesai. Periksa UI MLflow untuk detail lebih lanjut.")

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import start_http_server, Gauge, Counter
import joblib
import numpy as np
import uvicorn
import threading
import time

# Load model
model = joblib.load("models/best_model.joblib")

# Feature order must match training
FEATURE_NAMES = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']

# FastAPI app
app = FastAPI(title="AQI Classifier")

# Prometheus metrics (at least 10)
prediction_count = Counter("prediction_count", "Total predictions made")
model_latency = Gauge("model_latency_seconds", "Time to process prediction")
model_loaded = Gauge("model_loaded", "Model loaded successfully (1=Yes)")
input_feature_sum = Gauge("input_feature_sum", "Sum of all input features")
feature_average = Gauge("input_feature_avg", "Average input feature value")
last_prediction = Gauge("last_prediction_class", "Last predicted class (encoded as integer)")
min_feature_value = Gauge("min_input_feature", "Minimum input feature value")
max_feature_value = Gauge("max_input_feature", "Maximum input feature value")
predict_error = Counter("prediction_error_total", "Total prediction errors")

# Assume model loaded correctly
model_loaded.set(1)

# Pydantic schema
class AQIInput(BaseModel):
    CO: float
    Ozone: float
    NO2: float
    PM25: float

@app.post("/predict")
def predict(input_data: AQIInput):
    try:
        # Extract features in correct order
        features = [[
            input_data.CO,
            input_data.Ozone,
            input_data.NO2,
            input_data.PM25
        ]]
        
        np_features = np.array(features)

        # Log input stats
        input_feature_sum.set(np.sum(np_features))
        feature_average.set(np.mean(np_features))
        min_feature_value.set(np.min(np_features))
        max_feature_value.set(np.max(np_features))

        # Measure latency
        start_time = time.time()
        prediction = model.predict(np_features)[0]
        latency = time.time() - start_time
        model_latency.set(latency)

        # Update other metrics
        prediction_count.inc()
        last_prediction.set(hash(str(prediction)) % 1000)  # Convert class to numeric hash for simplicity

        return {"prediction": prediction}

    except Exception as e:
        predict_error.inc()
        return {"error": str(e)}

# Prometheus server (on port 9000)
def start_metrics_server():
    start_http_server(9000)

# Start Prometheus metrics on background thread
threading.Thread(target=start_metrics_server).start()

# Run FastAPI app (port 8000)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

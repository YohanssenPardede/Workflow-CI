from fastapi import FastAPI 
from pydantic import BaseModel 
from prometheus_client import start_http_server, Gauge, Counter
import joblib 
import pandas as pd 
import numpy as np
import uvicorn
import threading
import time

# Memuat Model Machine Learning
model = joblib.load("models/best_model.joblib")

# Definisi Fitur Model
FEATURE_NAMES = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="AQI Classifier")

# Definisi Metrik Prometheus
# Counter: Menghitung total prediksi yang telah dilakukan.
prediction_count = Counter("prediction_count", "Total predictions made")
# Gauge: Mengukur waktu yang dibutuhkan untuk memproses prediksi (dalam detik).
model_latency = Gauge("model_latency_seconds", "Time to process prediction")
# Gauge: Menunjukkan apakah model berhasil dimuat (1 = Ya, 0 = Tidak).
model_loaded = Gauge("model_loaded", "Model loaded successfully (1=Yes)")
# Gauge: Mengukur total atau jumlah nilai dari semua fitur input.
input_feature_sum = Gauge("input_feature_sum", "Sum of all input features")
# Gauge: Mengukur rata-rata nilai dari semua fitur input.
feature_average = Gauge("input_feature_avg", "Average input feature value")
# Gauge: Menyimpan kelas prediksi terakhir (di-encode sebagai integer).
last_prediction = Gauge("last_prediction_class", "Last predicted class (encoded as integer)")
# Gauge: Menyimpan nilai minimum dari fitur input.
min_feature_value = Gauge("min_input_feature", "Minimum input feature value")
# Gauge: Menyimpan nilai maksimum dari fitur input.
max_feature_value = Gauge("max_input_feature", "Maximum input feature value")
# Counter: Menghitung total kesalahan atau error saat melakukan prediksi.
predict_error = Counter("prediction_error_total", "Total prediction errors")

# Setelah model berhasil dimuat di awal skrip, set metrik 'model_loaded' menjadi 1.
model_loaded.set(1)

# Skema Data Input Pydantic
# Mendefinisikan skema data input yang diharapkan untuk endpoint prediksi.
class AQIInput(BaseModel):
    CO: float # Nilai CO (Carbon Monoxide) dalam bentuk float.
    Ozone: float # Nilai Ozone dalam bentuk float.
    NO2: float # Nilai NO2 (Nitrogen Dioxide) dalam bentuk float.
    PM25: float # Nilai PM2.5 (Particulate Matter 2.5) dalam bentuk float.

# Endpoint Prediksi API
# Mendefinisikan endpoint POST "/predict" yang akan menerima data AQIInput.
@app.post("/predict")
def predict(input_data: AQIInput):
    try:
        # Membuat Pandas DataFrame dari data input.
        # Penting: Urutan kolom harus sesuai dengan FEATURE_NAMES agar cocok dengan pelatihan model.
        features_df = pd.DataFrame([{
            "CO AQI Value": input_data.CO,
            "Ozone AQI Value": input_data.Ozone,
            "NO2 AQI Value": input_data.NO2,
            "PM2.5 AQI Value": input_data.PM25
        }])

        # Logging Metrik Input
        # Konversi DataFrame ke array NumPy untuk penghitungan metrik yang efisien.
        np_features = features_df.values
        # Memperbarui metrik 'input_feature_sum' dengan jumlah total semua fitur input.
        input_feature_sum.set(np.sum(np_features))
        # Memperbarui metrik 'feature_average' dengan rata-rata nilai fitur input.
        feature_average.set(np.mean(np_features))
        # Memperbarui metrik 'min_feature_value' dengan nilai minimum dari fitur input.
        min_feature_value.set(np.min(np_features))
        # Memperbarui metrik 'max_feature_value' dengan nilai maksimum dari fitur input.
        max_feature_value.set(np.max(np_features))

        # Melakukan Prediksi dan Mengukur Latensi
        start_time = time.time() # Mencatat waktu mulai sebelum prediksi.
        prediction = model.predict(features_df)[0] # Melakukan prediksi menggunakan model. [0] karena predict mengembalikan array.
        latency = time.time() - start_time # Menghitung latensi (waktu yang dihabiskan untuk prediksi).
        model_latency.set(latency) # Memperbarui metrik 'model_latency' dengan latensi yang diukur.

        # Memperbarui Metrik Prediksi
        prediction_count.inc() # Menaikkan hitungan metrik 'prediction_count' setiap kali prediksi berhasil.
        last_prediction.set(int(prediction)) # Memperbarui metrik 'last_prediction' dengan hasil prediksi.

        # Mengembalikan hasil prediksi dalam format JSON.
        return {"prediction": int(prediction)}

    except Exception as e:
        # Jika terjadi error selama proses prediksi, naikkan metrik 'predict_error'.
        predict_error.inc()
        # Mengembalikan pesan error dalam format JSON.
        return {"error": str(e)}

# Server Metrik Prometheus
# Fungsi untuk memulai server HTTP Prometheus yang akan mengekspos metrik.
def start_metrics_server():
    # Server Prometheus akan berjalan di port 9000 secara default.
    start_http_server(9000)

# Menjalankan Server Metrik di Latar Belakang
threading.Thread(target=start_metrics_server).start()

# Menjalankan Aplikasi FastAPI
# Blok ini akan dieksekusi hanya jika skrip dijalankan secara langsung (bukan diimpor sebagai modul).
if __name__ == "__main__":
    # Menjalankan aplikasi FastAPI menggunakan Uvicorn.
    # host="0.0.0.0" membuat aplikasi dapat diakses dari IP mana pun.
    # port=8000 adalah port di mana aplikasi FastAPI akan mendengarkan request.
    uvicorn.run(app, host="0.0.0.0", port=8000)

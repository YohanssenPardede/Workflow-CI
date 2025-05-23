# 🏭 AQI Model Training & Deployment CI/CD

Pipeline ini otomatis:
1. Melatih model klasifikasi AQI menggunakan MLflow Project  
2. Menyimpan artefak model ke dalam repo  
3. Membangun & mem-push Docker image ke Docker Hub  

## 📁 Struktur Direktori

```
Workflow-CI
├── MLProject/
│   ├── models
│   ├── Dockerfile             
│   ├── MLproject         
│   ├── Tautan ke Docker Hub.txt             
│   ├── aqi_preprocessing.csv
│   ├── conda.yml 
│   ├── modelling.py
│   ├── prometheus_exporter.py            
│   └── requirements.txt           
├── .github/
│   └── workflows/
│       └── main.yml   
├── aqi\_preprocessing.csv    
└── README.md          
````


## ⚙️ Setup Environment Lokal
1. **Install Miniconda**  
   Unduh & install dari https://www.anaconda.com/download

2. **Buat & aktifkan environment**  
   ```bash
   conda env create -f MLProject/conda.yml
   conda activate aqi-env
   ````

3. **Jalankan Training secara Manual**
   ```bash
   mlflow run MLProject/ \
       --env-manager local \
       --experiment-name AQI_Classification_Local
   ```

4. **Hasil**
   * Model tersimpan di `MLProject/models/`
   * Metadata & metrics dapat dilihat di `mlruns/` (jika lokal)

## 🤖 CI/CD dengan GitHub Actions
Workflow `.github/workflows/main.yml` (`Train AQI Model and Save to Repo`) berjalan pada tiap **push ke branch `main`** atau manual trigger:
1. **Checkout** kode
2. **Cache** dan **setup Conda** (`aqi-env`) menggunakan `MLProject/conda.yml`
3. **Jalankan** MLflow Project:
   ```bash
   mlflow run MLProject/ \
       --env-manager=local \
       --experiment-name=AQI_Classification_CI
   ```
4. **Commit & push** artefak model (`MLProject/models/`) kembali ke branch `main`
   * Menggunakan `GITHUB_TOKEN` agar tidak memerlukan cred eksternal
   * Commit message: `"Add trained model artifact [CI skip]"`

## 🐳 Docker Build & Push
Anda dapat mengakses Docker Hub untuk push image pada tautan [berikut](https://hub.docker.com/r/yohanssenpardede/aqi-model).
Setelah job `train` selesai, job `docker` akan:
1. **Checkout** kode
2. **Login** ke Docker Hub (via `DOCKERHUB_USERNAME` & `DOCKERHUB_TOKEN`)
3. **Build & push** image dari `MLProject/Dockerfile`
   * Tag `latest`
   * Tag SHA komit, misal: `yohanssenpardede/aqi-model:4f123ab`

```yaml
# contoh snippet build-push
uses: docker/build-push-action@v4
with:
  context: MLProject
  file: MLProject/Dockerfile
  push: true
  tags: |
    ${{ secrets.DOCKERHUB_USERNAME }}/aqi-model:latest
    ${{ secrets.DOCKERHUB_USERNAME }}/aqi-model:${{ github.sha }}
```

4. **Menjalankan Aplikasi (`CMD ["python", "prometheus_exporter.py"]`):**
`CMD ["python", "prometheus_exporter.py"]`: Ini adalah perintah default yang akan dijalankan saat kontainer Docker dimulai **jika tidak ada perintah lain yang ditentukan saat menjalankan kontainer**. Dengan perintah ini, `prometheus_exporter.py` akan dieksekusi menggunakan interpreter Python dari lingkungan Conda yang telah dibuat. Skrip ini akan memulai server HTTP di port 9000 (sesuai konvensi dan tujuan `prometheus_exporter.py`) dan mulai mengekspos metrik yang telah didefinisikan di dalamnya.

**Bagaimana `prometheus_exporter.py` di-*serving*:**
Ketika *image* Docker ini dibangun dan kemudian kontainer dijalankan (misalnya, dengan `docker run -p 9000:9000 <nama_image_docker>`), hal-hal berikut terjadi:
1. Kontainer dimulai, dan direktori `/app` menjadi direktori kerja.
2. Perintah `python prometheus_exporter.py` dieksekusi.
3. Skrip `prometheus_exporter.py` (yang seharusnya berisi kode untuk mengumpulkan metrik dan menjalankan server HTTP menggunakan pustaka seperti `prometheus_client`) akan memulai server di port yang ditentukan di dalam skrip itu sendiri (biasanya 9000).
4. Karena port 9000 diekspos (`EXPOSE 9000` di `Dockerfile`) dan diterbitkan (`-p 9000:9000` saat `docker run`), aplikasi Prometheus di luar kontainer dapat mengakses metrik dari kontainer ini dengan melakukan *scraping* (mengambil data) dari alamat IP kontainer di port 9000.

## 🔑 Secrets & Permissions
* **GITHUB\_TOKEN**:
  * Digunakan untuk commit & push artefak model tanpa login manual.
  * *Scope*: contents: write
* **DOCKERHUB\_USERNAME**, **DOCKERHUB\_TOKEN**: Untuk login & push Docker image.

## 📌 Catatan
* Pastikan file `aqi_preprocessing.csv` selalu tersedia di root sebelum menjalankan pipeline.
* Jika ingin melihat logs & metrics MLflow, jalankan secara lokal:
  ```bash
  mlflow ui
  ```
  dan buka `http://localhost:5000`.

# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement files
COPY MLProject/conda.yml ./conda.yml

# Install dependencies via conda (optional) or pip
RUN pip install mlflow scikit-learn pandas joblib

# Copy project code
COPY . .

# Expose port jika kamu ingin serve model
EXPOSE 5000

# Default command: jalankan script inference (kamu bisa ganti)
CMD ["python", "serve_model.py"]

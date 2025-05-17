# Gunakan image dasar Python
FROM python:3.12-slim

# Set direktori kerja dalam container
WORKDIR /app

# Salin file environment dan kode ke dalam container
COPY MLProject/ MLProject/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r MLProject/requirements.txt

# Jalankan skrip training
CMD ["python", "MLProject/train.py"]

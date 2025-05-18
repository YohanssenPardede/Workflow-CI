FROM continuumio/miniconda3

# Buat shell default agar bisa menggunakan conda environment
SHELL ["conda", "run", "-n", "aqi-env", "/bin/bash", "-c"]

# Salin environment YAML
COPY conda.yml ./conda.yml

# Install dependencies dan environment
RUN conda env create -f conda.yml && conda clean -afy

# Set environment path
ENV PATH=/opt/conda/envs/aqi-env/bin:$PATH

# Copy semua file project ke dalam image
COPY . /app
WORKDIR /app

# Expose port untuk FastAPI dan Prometheus
EXPOSE 8000
EXPOSE 9000

# Default command saat container dijalankan (bisa diubah saat docker run)
CMD ["python", "prometheus_exporter.py"]

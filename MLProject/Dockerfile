FROM continuumio/miniconda3

# Salin environment YAML
COPY conda.yml /tmp/conda.yml

# Buat environment dan bersihkan cache
RUN conda env create -f /tmp/conda.yml && conda clean -afy

# Tambahkan path environment ke PATH
ENV PATH=/opt/conda/envs/aqi-env/bin:$PATH

# Copy semua file project ke dalam image
COPY . /app
WORKDIR /app

# Expose port untuk FastAPI dan Prometheus (opsional)
EXPOSE 8000
EXPOSE 9000

# Default command saat container dijalankan
CMD ["python", "modelling.py"]

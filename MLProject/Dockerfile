FROM continuumio/miniconda3

# Copy environment file yang ada di folder MLProject (sama dengan Dockerfile)
COPY conda.yml ./conda.yml

# Install dependencies
RUN conda env create -f conda.yml

# Set environment
ENV PATH /opt/conda/envs/aqi-env/bin:$PATH

# Copy semua isi project (modelling.py dll) ke image
COPY . /app

WORKDIR /app

CMD ["python", "modelling.py"]

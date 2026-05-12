# Mistral-NeMo-8B 4-bit + LoRA TTT + vLLM
#
# Base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# - Includes Python 3.11, PyTorch 2.4.0, CUDA 12.4 runtime, cuDNN 9
# - Saves ~5min of pip install vs starting from nvidia/cuda
# - Sandbox build budget is 30min — we need every saving we can get.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HF_DATASETS_CACHE=/app/models

WORKDIR /app

# Minimal system deps (curl/git/build for any wheels that compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models && chmod 777 /app/models

# Python deps. torch is already in base image; pip will detect it and skip.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# App
COPY . /app/

CMD ["python3", "arc_main.py"]

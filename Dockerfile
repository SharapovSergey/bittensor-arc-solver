# Mistral-NeMo-8B 4-bit + LoRA TTT + vLLM
# Need CUDA runtime for unsloth/peft/bitsandbytes in prep phase.
# Base: NVIDIA CUDA 12.4 + Python 3.11.

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    HF_DATASETS_CACHE=/app/models

WORKDIR /app

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN mkdir -p /app/models && chmod 777 /app/models

# Python deps (CUDA-enabled torch + unsloth stack for TTT)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# App
COPY . /app/

CMD ["python3", "arc_main.py"]

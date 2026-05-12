FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    openai==1.35.0 \
    huggingface-hub==0.23.4 \
    requests==2.31.0

COPY . /app/

ENTRYPOINT ["python", "arc_main.py"]

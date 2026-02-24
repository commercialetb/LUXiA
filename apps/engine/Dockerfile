FROM python:3.11-slim

WORKDIR /app

# (opzionale ma consigliato per wheels/crypto/rustless builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render espone PORT (default 10000)
CMD ["sh", "-c", "uvicorn engine_api:app --host 0.0.0.0 --port ${PORT:-10000}"]

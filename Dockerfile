FROM python:3.12-slim

WORKDIR /app

# Install system deps for spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

# Cloud Run injects PORT (default 8080)
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}

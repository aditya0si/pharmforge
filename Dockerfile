FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY pharmforge pharmforge
COPY scripts scripts
COPY eval eval

# data/ is a runtime directory (feedback log, generated scripts); it is not
# tracked in git, so it is created here instead of being copied.
RUN mkdir -p data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "pharmforge.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

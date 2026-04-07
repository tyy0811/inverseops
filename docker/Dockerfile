FROM python:3.11-slim

WORKDIR /app

# Install system deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
# Minimal package stub so pip can resolve deps from pyproject.toml
COPY pyproject.toml README.md ./
RUN mkdir -p inverseops && touch inverseops/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf inverseops

# Copy application code
COPY inverseops/ inverseops/
COPY configs/ configs/

# Re-install with real code (deps already cached)
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "inverseops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

# System deps for soundfile / openSMILE / ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer-cached unless requirements.txt changes)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (Pitt/ and results/ are mounted at runtime)
COPY dementia_detection/ ./dementia_detection/
COPY config/             ./config/
COPY tests/              ./tests/
COPY pipeline.py         ./
COPY Makefile            ./

VOLUME ["/app/Pitt", "/app/results"]

CMD ["python", "-m", "pytest", "tests/", "-v"]

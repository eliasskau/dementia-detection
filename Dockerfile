FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

# Install system deps needed by librosa / soundfile / openSMILE
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer-cached unless they change)
COPY requirements.txt ./

# Create conda env with Python 3.10, then install packages via pip
RUN conda create -n dementia-detection python=3.10 -y && conda clean -afy
RUN conda run -n dementia-detection pip install --no-cache-dir -r requirements.txt

# Make conda env the default Python
ENV PATH="/opt/conda/envs/dementia-detection/bin:$PATH"
SHELL ["conda", "run", "-n", "dementia-detection", "/bin/bash", "-c"]

# Copy project code (data is mounted at runtime, not baked in)
COPY configs/  ./configs/
COPY src/      ./src/
COPY scripts/  ./scripts/
COPY tests/    ./tests/
COPY Makefile  ./

# Pitt data and results are mounted as volumes at runtime:
#   docker run -v /your/Pitt:/app/Pitt -v /your/results:/app/results ...
VOLUME ["/app/Pitt", "/app/results"]

# Default: run tests to verify the environment is correct
CMD ["conda", "run", "-n", "dementia-detection", "python", "-m", "pytest", "tests/", "-v"]

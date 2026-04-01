# 1. Start from NVIDIA's official CUDA development image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Set non-interactive timezone and install system packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    jq \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Set the working directory
WORKDIR /app

# 4. Copy dependency file and install
COPY requirements.txt .

# 5. Install PyTorch compiled specifically for CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install the rest of the MLOps stack
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Expose ports for MLflow tracking UI and potential FastAPI serving
EXPOSE 5000 8000

# 9. Default command (Can be overridden by GCP/Kaggle to run train.py)
CMD ["python", "-m", "src.models.train"]
# Base image: PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 from GitHub
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# Copy application code
COPY app/ ./app/

# Create directories
RUN mkdir -p checkpoints/sam2 outputs temp_frames

# Download SAM2 checkpoint from Cloudflare R2 at runtime via entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
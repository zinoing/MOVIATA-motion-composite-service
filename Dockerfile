# Base image: PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
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

# Patch hieradet.py: add weights_only=False to torch.load (required for PyTorch 2.x)
RUN python -c "\
import pathlib, re, sam2; \
p = pathlib.Path(sam2.__file__).parent / 'modeling/backbones/hieradet.py'; \
orig = p.read_text(); \
patched = re.sub(r'torch\.load\(f, map_location=\"cpu\"\)', 'torch.load(f, map_location=\"cpu\", weights_only=False)', orig); \
assert patched != orig, 'Patch did not match: check torch.load signature in hieradet.py'; \
p.write_text(patched); \
print('[patch] hieradet.py patched successfully')"

# Copy application code
COPY app/ ./app/

# Create directories
RUN mkdir -p checkpoints/sam2 outputs temp_frames

# Download SAM2 checkpoint from Cloudflare R2 at runtime via entrypoint
COPY entrypoint.sh handler.py ./
RUN chmod +x entrypoint.sh

EXPOSE 8000

# RunPod Serverless: handler.py boots FastAPI+Celery internally
# Direct deployment: ./entrypoint.sh
CMD ["python", "handler.py"]
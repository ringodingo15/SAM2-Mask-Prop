# GPU-ready Dockerfile (requires NVIDIA Container Toolkit)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Install SAM2 (adjust if you pin a commit)
RUN pip install "git+https://github.com/facebookresearch/sam2.git#egg=sam2"

COPY app /app/app
COPY web /app/web
COPY scripts /app/scripts
COPY docs /app/docs
COPY .env /app/.env || true

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cuda
# Set these to absolute paths at runtime or via bind mounts:
# ENV SAM2_MODEL_TYPE=sam2.1_hiera_large
# ENV SAM2_CHECKPOINT=/app/checkpoints/sam2.1_hiera_large.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
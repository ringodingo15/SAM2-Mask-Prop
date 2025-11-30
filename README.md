# SAM2 Mask Prop

A server-hosted web app for propagating object masks across a video using Meta's SAM 2.1 Large model, starting from Label Studio annotations (masks, points, rectangles). View overlays in-browser, and export generated masks as images.

Key features:
- Upload video or ZIP of frames (00000001.png, 00000002.png, ...)
- Upload Label Studio export JSON (masks, points, rectangle labels supported)
- Run mask propagation using SAM2.1 Large
- Visualize translucent overlays in browser
- Export resulting masks as images (PNG)

This app is intended for on-prem/server deployment within a network. GPU acceleration is recommended.

---

## Table of Contents
- [Requirements](#requirements)
- [Quick Start (Python)](#quick-start-python)
- [Download SAM2.1 Large Weights](#download-sam21-large-weights)
- [Running the App](#running-the-app)
- [Using the Web UI](#using-the-web-ui)
- [Label Studio Export Guide](#label-studio-export-guide)
- [Data Layout](#data-layout)
- [Exporting Masks](#exporting-masks)
- [Docker (GPU)](#docker-gpu)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Requirements
- OS: Linux (Ubuntu 20.04+/22.04 recommended). macOS may work for CPU dev/testing.
- Python: 3.10 or 3.11 recommended.
- GPU: NVIDIA GPU recommended with CUDA 12.x for best performance.
- Drivers: NVIDIA drivers and CUDA toolkit installed for GPU usage.
- Disk: Sufficient space for model weights and generated masks.

Python dependencies (installed via `requirements.txt`):
- torch (with CUDA build if using GPU)
- torchvision
- fastapi, uvicorn
- opencv-python, numpy, pillow
- pydantic, python-multipart, aiofiles
- pydantic-settings
- tqdm
- starlette
- (SAM2) from source or pip if available

---

## Quick Start (Python)

1) Clone repository
```
git clone https://github.com/your-org/SAM2-Mask-Prop.git
cd SAM2-Mask-Prop
```

2) Create virtual environment and install dependencies
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Install SAM2 from source (recommended)
- Repo: facebookresearch/sam2
- Follow their README for installation. Typically:
```
pip install "git+https://github.com/facebookresearch/sam2.git#egg=sam2"
```
If you build from source with CUDA ops, follow their instructions for your environment.

4) Download SAM2.1 Large weights
- See [Download SAM2.1 Large Weights](#download-sam21-large-weights)

5) Set environment variables (or use `.env`)
```
export SAM2_MODEL_TYPE="sam2.1_hiera_large"
export SAM2_CHECKPOINT="/path/to/checkpoints/sam2.1_hiera_large.pt"
export DEVICE="cuda"   # or "cpu"
```

6) Run the server
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

7) Open the web UI
- Navigate to: http://<server-ip>:8000

---

## Download SAM2.1 Large Weights

This repo provides a helper script that uses documented sources to download the SAM2.1 Large checkpoint. If URLs change, refer to the official [facebookresearch/sam2](https://github.com/facebookresearch/sam2) repository for the latest release assets.

Run:
```
python scripts/download_sam2_weights.py --model sam2.1_hiera_large --out ./checkpoints
```

Then set:
```
export SAM2_MODEL_TYPE="sam2.1_hiera_large"
export SAM2_CHECKPOINT="$(pwd)/checkpoints/sam2.1_hiera_large.pt"
```

If the script fails due to changed URLs, download the checkpoint manually from the SAM2 GitHub release page and place it in `./checkpoints`.

---

## Running the App

- Development:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Production (example with uvicorn workers):
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Ensure your firewall/security rules allow access from your LAN if you want network users to access it.

---

## Using the Web UI

1) Upload a video or a frames ZIP
- Video: MP4/MOV/AVI. The server extracts frames internally.
- Frames ZIP: Containing images named with zero-padded frame indices, e.g. `00000001.png`, `00000002.png`, ...
  - Use PNG or JPG. PNG is recommended for consistent downstream masking.

2) Upload Label Studio export JSON
- See [Label Studio Export Guide](#label-studio-export-guide)
- Supported label types:
  - Mask (brush/rle-based where available)
  - Point
  - Rectangle (RectangleLabels)
- The app parses the export, maps annotated frames to prompts.

3) Run Propagation
- The server spawns a background job that:
  - Loads the SAM2.1 Large model
  - Adds prompts for relevant frames
  - Propagates object masks across the video’s frames
- Check progress in the UI; when complete, mask images are available for visualization/export.

4) Visualize Overlays
- Use the built-in video/frames viewer. A translucent red overlay shows the mask(s).

5) Export Masks
- Download all generated masks as a ZIP from the UI.

---

## Label Studio Export Guide

This app expects the raw JSON export from Label Studio. A typical workflow:

1) In Label Studio, create a Video project.
   - You can choose a template supporting:
     - RectangleLabels for bounding boxes
     - KeyPointLabels or Points for point prompts
     - BrushLabels for pixel masks (segmentation)

2) Annotate one or more frames. Make sure:
   - Annotations correspond to specific frames (Label Studio records frame index/time).
   - Use consistent labels for each object you want to track/propagate.

3) Export
   - In the project, click Export.
   - Choose JSON export.
   - Save the exported `.json` file.

4) Upload to this app
   - Use the "Upload Label Studio JSON" button.
   - The server maps frame indices and prompt types to SAM2 prompts.

Notes:
- RectangleLabels: parsed as bounding boxes (x, y, width, height; percentage coords are converted using frame dimensions).
- Points/KeyPoints: parsed as positive clicks (x, y).
- Masks/BrushLabels: uses RLE or polygon data if present; converted to binary mask prompts.
- If you export multiple tasks in one JSON, the app uses the first relevant video task by default (or you can choose via UI when multiple tasks match).

For a concrete example of JSON structure, see [docs/LabelStudio.md](docs/LabelStudio.md).

---

## Data Layout

By default, the app stores data under `./data`:
- uploads/
  - jobs/<job_id>/
    - video/ (original video)
    - frames/ (extracted frames; or from uploaded ZIP)
    - labelstudio/ (uploaded export JSON)
- outputs/
  - jobs/<job_id>/
    - masks/ (generated per-frame masks, PNG)
    - overlays/ (optionally generated visualization)
    - export.zip (on demand)

You can customize paths via environment variables in `.env` or `app/config.py`.

---

## Exporting Masks

- Use the “Export Masks” button after propagation completes.
- The server returns a ZIP containing per-frame masks:
  - Filenames match the frame numbering convention, e.g. `00000001.png`, etc.
  - Format: 8-bit single-channel PNG (0 = background, 255 = object) by default.
  - If multiple objects/classes are present, masks are saved per object id and/or composited; see configuration.

---

## Docker (GPU)

1) Install NVIDIA Container Toolkit:
   - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

2) Build and run:
```
docker compose up --build
```

3) Access the app at:
- http://localhost:8000

Use environment variables in `docker-compose.yml` or a `.env` file to set `SAM2_MODEL_TYPE`, `SAM2_CHECKPOINT`, and `DEVICE=cuda`.

---

## Configuration

You can set these via environment variables or `.env`:
- SAM2_MODEL_TYPE: default `sam2.1_hiera_large`
- SAM2_CHECKPOINT: path to the checkpoint file
- DEVICE: `cuda` or `cpu` (default attempts `cuda` if available)
- DATA_ROOT: base data directory (default `./data`)
- MAX_WORKERS: background workers (default 1-2)
- MASK_OUTPUT_MODE: `single` or `per_label` for multiple object handling
- FRAME_EXT: `png` by default

See `app/config.py` for full list and defaults.

---

## Troubleshooting

- SAM2 import errors:
  - Ensure `sam2` is installed from the official repo.
  - Check CUDA toolkit versions match your PyTorch build.
  - Try CPU mode by setting `DEVICE=cpu` to isolate GPU issues.
- Checkpoint not found:
  - Verify `SAM2_CHECKPOINT` path.
- Label Studio mapping issues:
  - Inspect your export JSON.
  - Confirm frame indices align with extracted frames.
  - See [docs/LabelStudio.md](docs/LabelStudio.md) for examples.
- Performance:
  - Use GPU.
  - Avoid very large frame resolutions; consider downscaling before processing.
- Long videos:
  - Monitor server RAM/VRAM.
  - Consider chunking or processing segments.

---

## Known Limitations

- The app is designed around the SAM2.1 video prediction API; SAM2 package API may evolve. If import names differ, refer to `app/sam2_infer.py` and the official repo to adjust.
- Multi-object handling: By default, masks are composited; per-label mask export is supported via config.
- Not a Label Studio plugin—data flows via JSON export/import.

---

## License

MIT License. See [LICENSE](LICENSE).

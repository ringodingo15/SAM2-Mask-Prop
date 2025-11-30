# SAM2 Mask Prop (Local-first)

A server-hosted web app (FastAPI) for propagating object masks across a video using Meta's SAM 2.1 Large model, starting from Label Studio annotations (masks, points, rectangles). View overlays in-browser, and export generated masks as images.

Local-first:
- No Docker required. Use venv and `.env` in this repo.
- All configuration is loaded from `.env` via `pydantic-settings`.
- Docker files are included only for optional use; you can ignore them.

Key features:
- Upload video or ZIP of frames (00000001.png, 00000002.png, ...)
- Upload Label Studio export JSON (masks, points, rectangles supported)
- Run mask propagation using SAM2.1 Large
- Visualize translucent overlays in browser
- Export resulting masks as images (PNG)

GPU acceleration is recommended but not required. CPU mode works for small tests.

---

## Requirements

- OS: Linux (Ubuntu 20.04+/22.04 recommended). macOS may work for CPU dev/testing.
- Python: 3.10 or 3.11 recommended.
- GPU: NVIDIA GPU recommended (CUDA 12.x). If not available, set `DEVICE=cpu` in `.env`.
- Disk: Sufficient space for model weights and generated masks.

Python packages: see `requirements.txt`.

---

## Local Quick Start (no Docker)

1) Clone the repo
```
git clone https://github.com/ringodingo15/SAM2-Mask-Prop.git
cd SAM2-Mask-Prop
git checkout -b init-app-local
```

2) Create a virtual environment and install deps
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Install SAM2 locally (from source)
SAM2 is evolving; install directly from the official GitHub:
```
pip install "git+https://github.com/facebookresearch/sam2.git#egg=sam2"
```
If you need CUDA-optimized ops per their README, follow SAM2’s installation guide and ensure your PyTorch build matches your CUDA version.

4) Download SAM2.1 Large weights
Use the script provided (or download manually from SAM2 releases):
```
python scripts/download_sam2_weights.py --model sam2.1_hiera_large --out ./checkpoints
```
If URLs change, grab the checkpoint from SAM2’s release page and place it at:
```
./checkpoints/sam2.1_hiera_large.pt
```

5) Configure local environment
Edit `.env` (already in repo). Minimal example:
```
SAM2_MODEL_TYPE=sam2.1_hiera_large
SAM2_CHECKPOINT=./checkpoints/sam2.1_hiera_large.pt
DEVICE=cuda      # set to "cpu" if you don’t have GPU
DATA_ROOT=./data
FRAME_EXT=png
```
Note: You do NOT need to export shell variables; the app reads `.env` automatically.

6) Run the server locally
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Now open:
- http://localhost:8000

---

## Using the Web UI

1) Create a new job
- Button: “Create New Job” → displays Job ID

2) Upload video OR frames ZIP
- Video: MP4/MOV/AVI. Frames extracted to `./data/uploads/jobs/<job_id>/frames`.
- Frames ZIP: images named with zero-padded indices (e.g., `00000001.png`).

3) Upload Label Studio export JSON
- Button: “Upload LS Export” → selects the JSON export.
- See [docs/LabelStudio.md](docs/LabelStudio.md) for format details.

4) Run Propagation
- Option: labels mode (composite or per_label)
- Button: “Start Propagation” → tracks status until completed.

5) Preview and Export
- Viewer shows frame with translucent overlay.
- Click “Download Masks ZIP” to get all masks as PNG.

---

## Label Studio Export Guide

- We support RectangleLabels (boxes), Points (keypoints), and BrushLabels (pixel masks).
- Coordinate conversion from percent to pixels uses `original_width` and `original_height`.
- Masks decode RLE where available; see docs for details.

See [docs/LabelStudio.md](docs/LabelStudio.md).

---

## Data Layout (Local)

Default paths under `DATA_ROOT` from `.env` (default `./data`):
- uploads/jobs/<job_id>/video/
- uploads/jobs/<job_id>/frames/
- uploads/jobs/<job_id>/labelstudio/
- outputs/jobs/<job_id>/masks/
- outputs/jobs/<job_id>/overlays/
- outputs/jobs/<job_id>/export.zip

---

## Exporting Masks

- Export ZIP contains mask images (PNG).
- Filenames match the source frame numbers (e.g., `00000001.png`).
- Default masks are single-channel: 0 = background, 255 = object.

---

## Optional: Local helper scripts

- `scripts/download_sam2_weights.py` downloads weights into `./checkpoints`.
- `scripts/local_setup.sh` bootstraps the environment end-to-end (venv, pip install, SAM2 install, weights download).

Run:
```
bash scripts/local_setup.sh
```

---

## Optional: Docker (IGNORE if you don’t use it)

Docker and docker-compose files are included for environments using containers. You do not need to touch them; the app runs locally as described above.

---

## Troubleshooting (Local)

- SAM2 import errors:
  - Ensure `sam2` is installed: `pip list | grep sam2`.
  - If CUDA build mismatch, try `DEVICE=cpu` in `.env` to validate flow.
- Checkpoint not found:
  - Verify `.env` has `SAM2_CHECKPOINT` pointing to the `.pt` file.
- Label Studio parsing:
  - Confirm JSON structure and `frame` indices.
  - Frame naming: ensure zero-padded names (the app tries to normalize).
- Performance:
  - Use GPU for long videos.
  - Downscale frames prior to upload if you hit memory limits.

---

## Notes

- `app/sam2_infer.py` contains a safe fallback (per-frame heuristic). Replace it with actual SAM2 video propagation API calls per the installed SAM2 package’s version.
- As SAM2 APIs may evolve, adjust the imports in `sam2_infer.py` to fit the installed version from facebookresearch/sam2.

---

## License

MIT License. See [LICENSE](LICENSE).
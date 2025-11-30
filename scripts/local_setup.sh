#!/usr/bin/env bash
set -euo pipefail

# Local-first bootstrap script
# - Creates venv
# - Installs requirements
# - Installs SAM2 from source
# - Downloads SAM2.1 Large checkpoint to ./checkpoints

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAM2_MODEL="${SAM2_MODEL:-sam2.1_hiera_large}"

echo "[*] Creating virtual environment..."
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing requirements..."
pip install -r requirements.txt

echo "[*] Installing SAM2 from source..."
pip install "git+https://github.com/facebookresearch/sam2.git#egg=sam2"

echo "[*] Downloading SAM2 weights ($SAM2_MODEL)..."
python scripts/download_sam2_weights.py --model "$SAM2_MODEL" --out ./checkpoints

echo "[*] Done. Configure .env if needed, then run:"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8000"
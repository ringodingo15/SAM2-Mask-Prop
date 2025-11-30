import os
import io
import json
import shutil
import zipfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from app.config import settings
from app.video_utils import extract_frames_from_video, validate_frame_zip, ensure_zero_padded_names
from app.labelstudio_parser import parse_labelstudio_export
from app.sam2_infer import SAM2VideoPropagator, PropagationResult
from app.progress import JobManager

app = FastAPI(title="SAM2 Mask Prop", version="1.0.0")

# CORS for LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
app.mount("/", StaticFiles(directory="web", html=True), name="web")

DATA_ROOT = Path(settings.DATA_ROOT).resolve()
UPLOADS = DATA_ROOT / "uploads" / "jobs"
OUTPUTS = DATA_ROOT / "outputs" / "jobs"
CHECKPOINT = Path(settings.SAM2_CHECKPOINT) if settings.SAM2_CHECKPOINT else None

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

jobs = JobManager()


def _job_paths(job_id: str) -> Dict[str, Path]:
    job_root_u = UPLOADS / job_id
    job_root_o = OUTPUTS / job_id
    return dict(
        job_id=job_id,
        upload_root=job_root_u,
        output_root=job_root_o,
        video_dir=job_root_u / "video",
        frames_dir=job_root_u / "frames",
        ls_dir=job_root_u / "labelstudio",
        masks_dir=job_root_o / "masks",
        overlays_dir=job_root_o / "overlays",
        export_zip=job_root_o / "export.zip",
    )


@app.post("/api/new_job")
def new_job():
    job_id = uuid.uuid4().hex[:8]
    p = _job_paths(job_id)
    for k, d in p.items():
        if k.endswith("_dir") or k.endswith("_root"):
            Path(d).mkdir(parents=True, exist_ok=True)
    jobs.create(job_id)
    return {"job_id": job_id}


@app.post("/api/upload_video")
async def upload_video(job_id: str = Form(...), file: UploadFile = File(...)):
    p = _job_paths(job_id)
    if not (UPLOADS / job_id).exists():
        raise HTTPException(400, "Invalid job_id. Create a job first.")

    # Save video file
    video_path = p["video_dir"] / file.filename
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract frames
    frames_dir = p["frames_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = extract_frames_from_video(str(video_path), str(frames_dir), ext=settings.FRAME_EXT)
    if count == 0:
        raise HTTPException(400, "Failed to extract frames from video.")

    ensure_zero_padded_names(frames_dir)

    return {"message": "Video uploaded and frames extracted.", "frame_count": count}


@app.post("/api/upload_frames_zip")
async def upload_frames_zip(job_id: str = Form(...), file: UploadFile = File(...)):
    p = _job_paths(job_id)
    if not (UPLOADS / job_id).exists():
        raise HTTPException(400, "Invalid job_id. Create a job first.")

    frames_dir = p["frames_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = frames_dir / "frames.zip"
    with open(tmp_zip, "wb") as f:
        f.write(await file.read())

    # Extract ZIP
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(frames_dir)
    tmp_zip.unlink(missing_ok=True)

    # Validate frame files
    count = validate_frame_zip(frames_dir, ext=settings.FRAME_EXT)
    if count == 0:
        raise HTTPException(400, "No frames detected in ZIP. Expected images with zero-padded names.")

    ensure_zero_padded_names(frames_dir)
    return {"message": "Frames ZIP uploaded and extracted.", "frame_count": count}


@app.post("/api/upload_labelstudio")
async def upload_labelstudio(job_id: str = Form(...), file: UploadFile = File(...)):
    p = _job_paths(job_id)
    if not (UPLOADS / job_id).exists():
        raise HTTPException(400, "Invalid job_id. Create a job first.")

    ls_dir = p["ls_dir"]
    ls_dir.mkdir(parents=True, exist_ok=True)
    ls_path = ls_dir / file.filename
    with open(ls_path, "wb") as f:
        f.write(await file.read())

    # Light validation
    try:
        with open(ls_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected a list of tasks.")
    except Exception as e:
        raise HTTPException(400, f"Invalid Label Studio JSON: {e}")

    return {"message": "Label Studio export uploaded."}


@app.post("/api/propagate")
def propagate(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    labels_mode: str = Form("composite"),  # or 'per_label'
):
    p = _job_paths(job_id)
    frames_dir: Path = p["frames_dir"]
    ls_dir: Path = p["ls_dir"]
    masks_dir: Path = p["masks_dir"]
    overlays_dir: Path = p["overlays_dir"]
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Check inputs
    if not frames_dir.exists() or len(list(frames_dir.glob(f"*.{settings.FRAME_EXT}"))) == 0:
        raise HTTPException(400, "No frames found. Upload a video or frames ZIP first.")
    ls_files = list(ls_dir.glob("*.json"))
    if not ls_files:
        raise HTTPException(400, "No Label Studio export found. Upload a JSON export first.")

    # Parse Label Studio prompts
    try:
        prompts = parse_labelstudio_export(
            ls_json_path=str(ls_files[0]),
            frames_dir=str(frames_dir),
            frame_ext=settings.FRAME_EXT,
        )
        if prompts.is_empty():
            raise ValueError("No usable prompts found in Label Studio export.")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse Label Studio export: {e}")

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(400, "Invalid job_id.")

    # Run propagation in background
    def task():
        try:
            jobs.update(job_id, status="running", progress=0, message="Initializing SAM2 model...")
            propagator = SAM2VideoPropagator(
                model_type=settings.SAM2_MODEL_TYPE,
                checkpoint_path=str(CHECKPOINT) if CHECKPOINT else "",
                device=settings.DEVICE,
            )
            jobs.update(job_id, message="Loading frames...")
            frame_paths = sorted(frames_dir.glob(f"*.{settings.FRAME_EXT}"))
            result: PropagationResult = propagator.propagate(
                frame_paths=frame_paths,
                prompts=prompts,
                labels_mode=labels_mode,
                progress_cb=lambda p, msg=None: jobs.update(job_id, progress=p, message=msg or ""),
                output_masks_dir=str(masks_dir),
                output_overlays_dir=str(overlays_dir),
            )
            jobs.update(job_id, status="completed", progress=100, message="Propagation complete.", meta=dict(
                frame_count=len(frame_paths),
                objects=len(result.object_labels),
            ))
        except Exception as e:
            jobs.update(job_id, status="failed", message=str(e))

    background_tasks.add_task(task)
    jobs.update(job_id, status="queued", progress=0, message="Queued")
    return {"message": "Propagation started.", "job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/frames/{job_id}/list")
def list_frames(job_id: str):
    p = _job_paths(job_id)
    frames_dir: Path = p["frames_dir"]
    if not frames_dir.exists():
        raise HTTPException(404, "Frames not found.")
    frames = [f"/data/{job_id}/frames/{fp.name}" for fp in sorted(frames_dir.glob(f"*.{settings.FRAME_EXT}"))]
    return {"frames": frames}


@app.get("/api/masks/{job_id}/list")
def list_masks(job_id: str):
    p = _job_paths(job_id)
    masks_dir: Path = p["masks_dir"]
    if not masks_dir.exists():
        raise HTTPException(404, "Masks not found.")
    masks = [f"/data/{job_id}/masks/{fp.name}" for fp in sorted(masks_dir.glob("*.png"))]
    return {"masks": masks}


@app.get("/api/export/{job_id}")
def export_masks(job_id: str):
    p = _job_paths(job_id)
    masks_dir: Path = p["masks_dir"]
    export_zip: Path = p["export_zip"]
    if not masks_dir.exists():
        raise HTTPException(404, "Masks not found.")
    # Zip masks
    with zipfile.ZipFile(export_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(masks_dir.glob("*.png")):
            zf.write(fp, arcname=fp.name)
    return FileResponse(export_zip, filename=f"{job_id}_masks.zip")


# Static data access for frames and masks (served under /data/{job_id}/...)
@app.get("/data/{job_id}/frames/{filename}")
def serve_frame(job_id: str, filename: str):
    p = _job_paths(job_id)
    path = p["frames_dir"] / filename
    if not path.exists():
        raise HTTPException(404, "Frame not found.")
    return FileResponse(path)


@app.get("/data/{job_id}/masks/{filename}")
def serve_mask(job_id: str, filename: str):
    p = _job_paths(job_id)
    path = p["masks_dir"] / filename
    if not path.exists():
        raise HTTPException(404, "Mask not found.")
    return FileResponse(path)
from pathlib import Path
from typing import Optional
import cv2
import os
import re

def extract_frames_from_video(video_path: str, out_dir: str, ext: str = "png") -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idx = 1
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = f"{idx:08d}.{ext}"
        cv2.imwrite(str(Path(out_dir) / fname), frame)
        idx += 1
        count += 1
    cap.release()
    return count

def validate_frame_zip(frames_dir: Path, ext: str = "png") -> int:
    frames = list(frames_dir.rglob(f"*.{ext}"))
    # If zip contains nested folders, move images up
    if frames and not all(f.parent == frames_dir for f in frames):
        for f in frames:
            dest = frames_dir / f.name
            if not dest.exists():
                f.replace(dest)
    frames = sorted(frames_dir.glob(f"*.{ext}"))
    return len(frames)

def ensure_zero_padded_names(frames_dir: Path):
    # Rename files to 8-digit zero padded if needed
    pattern = re.compile(r"(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    for fp in sorted(frames_dir.iterdir()):
        m = pattern.match(fp.name)
        if not m:
            # Try to extract digits from name
            digits = "".join([c for c in fp.stem if c.isdigit()])
            if digits:
                ext = fp.suffix.lower().lstrip(".")
                new = frames_dir / f"{int(digits):08d}.{ext}"
                if new != fp:
                    fp.rename(new)
        else:
            num, ext = m.group(1), m.group(2)
            new = frames_dir / f"{int(num):08d}.{ext.lower()}"
            if new != fp:
                fp.rename(new)
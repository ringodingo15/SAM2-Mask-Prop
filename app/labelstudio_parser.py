import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

@dataclass
class BoxPrompt:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str

@dataclass
class PointPrompt:
    frame: int
    x: float
    y: float
    label: str
    positive: bool = True

@dataclass
class MaskPrompt:
    frame: int
    mask: np.ndarray  # HxW bool or uint8
    label: str

@dataclass
class ParsedPrompts:
    boxes: List[BoxPrompt] = field(default_factory=list)
    points: List[PointPrompt] = field(default_factory=list)
    masks: List[MaskPrompt] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.boxes or self.points or self.masks)

def _percent_to_abs(val: float, size: int) -> float:
    return (val / 100.0) * size

def _decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    # Basic RLE decoder (Label Studio style may differ; adjust if needed)
    # Expecting "counts" in COCO-like format is typical; many LS exports provide custom rle string.
    # This is a placeholder simple decoder for "num:len" pairs. For robustness, integrate pycocotools if available.
    try:
        import pycocotools.mask as mask_utils
        # If rle is json-serializable dict with counts/size
        # Try parse as JSON
        try:
            obj = json.loads(rle)
            return mask_utils.decode(obj).astype(bool)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback simple decoder: space-separated integers alt format "counts" (not guaranteed)
    arr = np.zeros((height * width,), dtype=np.uint8)
    idx = 0
    try:
        parts = [int(x) for x in rle.strip().split()]
        val = 0
        for run in parts:
            if val == 1:
                arr[idx:idx+run] = 1
            idx += run
            val = 1 - val
        return arr.reshape((height, width)).astype(bool)
    except Exception:
        # If we cannot decode, return empty mask
        return np.zeros((height, width), dtype=bool)

def parse_labelstudio_export(ls_json_path: str, frames_dir: str, frame_ext: str = "png") -> ParsedPrompts:
    with open(ls_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = sorted(Path(frames_dir).glob(f"*.{frame_ext}"))
    if not frames:
        raise ValueError("No frames found in frames_dir.")

    prompts = ParsedPrompts()
    # Strategy: take first task with annotations
    task = None
    for t in data:
        if t.get("annotations"):
            task = t
            break
    if task is None:
        return prompts

    results = []
    for ann in task.get("annotations", []):
        for r in ann.get("result", []):
            results.append(r)

    # We assume uniform resolution across frames
    # Attempt to fetch from any result: original_width/height
    W = H = None
    for r in results:
        ow = r.get("original_width")
        oh = r.get("original_height")
        if ow and oh:
            W, H = int(ow), int(oh)
            break
    if W is None or H is None:
        # Fallback: read first frame
        import cv2
        img = cv2.imread(str(frames[0]))
        H, W = img.shape[:2]

    def clamp_frame_index(fr: int) -> int:
        # Normalize frames: many exports are 0-based; our file names start at 1-based.
        # If out of range, clamp.
        if fr <= 0:
            return 1
        if fr > len(frames):
            return len(frames)
        return fr

    for r in results:
        rtype = r.get("type")
        val = r.get("value", {})
        label = None
        fr = val.get("frame")
        if fr is None:
            # Try to derive frame from time? Not implemented.
            continue
        frame_idx = clamp_frame_index(int(fr) + 1)  # convert 0-based to 1-based

        if rtype in ("rectanglelabels", "rectangleregions"):
            rect_labels = val.get("rectanglelabels") or val.get("labels") or []
            label = rect_labels[0] if rect_labels else "object"
            x = _percent_to_abs(val.get("x", 0.0), W)
            y = _percent_to_abs(val.get("y", 0.0), H)
            w = _percent_to_abs(val.get("width", 0.0), W)
            h = _percent_to_abs(val.get("height", 0.0), H)
            prompts.boxes.append(BoxPrompt(frame=frame_idx, x1=x, y1=y, x2=x+w, y2=y+h, label=label))

        elif rtype in ("keypointlabels", "keypoints", "pointlabels"):
            pt_labels = val.get("keypointlabels") or val.get("labels") or []
            label = pt_labels[0] if pt_labels else "object"
            x = _percent_to_abs(val.get("x", 0.0), W)
            y = _percent_to_abs(val.get("y", 0.0), H)
            prompts.points.append(PointPrompt(frame=frame_idx, x=x, y=y, label=label, positive=True))

        elif rtype in ("brushlabels", "masklabels", "brush"):
            mask_labels = val.get("brushlabels") or val.get("labels") or []
            label = mask_labels[0] if mask_labels else "object"
            # RLE decode
            rle = val.get("rle")
            if rle:
                mask = _decode_rle(rle, height=H, width=W)
                prompts.masks.append(MaskPrompt(frame=frame_idx, mask=mask.astype(np.uint8), label=label))
            else:
                # TODO: polygon to raster if provided
                pass

    return prompts
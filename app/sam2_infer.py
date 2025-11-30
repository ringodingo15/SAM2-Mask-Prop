from dataclasses import dataclass, field
from typing import List, Callable, Iterable, Dict
from pathlib import Path
import numpy as np
import cv2
import torch

from app.labelstudio_parser import ParsedPrompts, BoxPrompt, PointPrompt, MaskPrompt

ProgressCB = Callable[[int, str], None]

@dataclass
class PropagationResult:
    object_labels: List[str] = field(default_factory=list)

class SAM2VideoPropagator:
    def __init__(self, model_type: str, checkpoint_path: str, device: str = "cuda"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.predictor = self._load_predictor()

    def _load_predictor(self):
        """
        Loads the SAM2 video predictor.

        Note: The exact API may differ based on SAM2 releases. Adjust imports/methods accordingly
        to match the facebookresearch/sam2 repo version you install.

        Common patterns in SAM2 repos include something like:
            from sam2.build_sam import build_sam2
            from sam2.video_predictor import Sam2VideoPredictor

        Fallback: We raise with a helpful message if imports are missing.
        """
        try:
            # Example import flow; adjust to actual public API of the installed SAM2 package.
            # These names may differ; consult the installed package.
            from sam2.build_sam import build_sam2
            from sam2.video_predictor import SAM2VideoPredictor  # hypothetical
        except Exception:
            # Attempt alternative names
            try:
                from sam2.build_sam import build_sam2_video_predictor as build_sam2
                from sam2.video_predictor import SAM2VideoPredictor  # hypothetical
            except Exception:
                raise ImportError(
                    "Could not import SAM2 video predictor. Please check the installed 'sam2' package "
                    "and adjust imports in app/sam2_infer.py to match the current API. "
                    "Refer to https://github.com/facebookresearch/sam2"
                )

        # Build model (example; adjust args to your installed SAM2)
        model = build_sam2(model_type=self.model_type, checkpoint=self.checkpoint_path)
        predictor = SAM2VideoPredictor(model, device=self.device)
        return predictor

    def _draw_overlay(self, image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4) -> np.ndarray:
        overlay = image.copy()
        red = np.zeros_like(image)
        red[..., 2] = 255
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * red[mask_bool]
        return overlay

    def propagate(
        self,
        frame_paths: Iterable[Path],
        prompts: ParsedPrompts,
        labels_mode: str,
        progress_cb: ProgressCB,
        output_masks_dir: str,
        output_overlays_dir: str,
    ) -> PropagationResult:
        frame_paths = list(frame_paths)
        H, W = None, None
        if not frame_paths:
            raise ValueError("No frames to process.")

        # Read first frame shape
        img0 = cv2.imread(str(frame_paths[0]))
        H, W = img0.shape[:2]

        # Group prompts by label/object
        # This allows multi-object propagation if the predictor API supports it.
        by_label: Dict[str, Dict[str, List]] = {}
        for b in prompts.boxes:
            by_label.setdefault(b.label, {}).setdefault("boxes", []).append(b)
        for p in prompts.points:
            by_label.setdefault(p.label, {}).setdefault("points", []).append(p)
        for m in prompts.masks:
            by_label.setdefault(m.label, {}).setdefault("masks", []).append(m)

        obj_labels = list(by_label.keys()) if by_label else ["object"]

        # Pseudo-code: Add prompts for each labeled frame, then run propagation.
        # Actual SAM2 API may differ. Replace the following pseudo with the correct calls.

        # Prepare output dirs
        out_masks_dir = Path(output_masks_dir)
        out_overlays_dir = Path(output_overlays_dir)
        out_masks_dir.mkdir(parents=True, exist_ok=True)
        out_overlays_dir.mkdir(parents=True, exist_ok=True)

        total = len(frame_paths)
        # Placeholder mask accumulator (for simple single-object scenario)
        # For multi-object, you'd compose or save per-object.
        accumulated_masks = [np.zeros((H, W), dtype=np.uint8) for _ in frame_paths]

        # Add prompts to predictor (example-style, to be adapted to API)
        try:
            self.predictor.reset()
        except Exception:
            pass

        # The following block is conceptual; replace with actual predictor methods
        # when integrating with the real SAM2 video API.
        # For each label/object, add prompts at their respective frames.
        # Then call something like predictor.propagate(frame_paths) -> returns masks per frame per object.

        # Begin naive loop fallback (if no API ready): per-frame SAM segmentation using box/point/mask on that frame only.
        # This is NOT true temporal propagation, but serves as a safe fallback structure.
        # Replace this with the actual video propagation call for production use.
        for i, fp in enumerate(frame_paths):
            img = cv2.imread(str(fp))
            frame_index_1based = i + 1

            # Select prompts that belong to this frame (1-based)
            frame_boxes = [b for b in prompts.boxes if b.frame == frame_index_1based]
            frame_points = [p for p in prompts.points if p.frame == frame_index_1based]
            frame_masks = [m for m in prompts.masks if m.frame == frame_index_1based]

            # TODO: Replace with predictor.add_box/frame, predictor.add_point/frame, predictor.add_mask/frame style calls.
            # For now, compose a heuristic mask from provided prompts:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # Convert boxes to masks
            for b in frame_boxes:
                x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])
                mask[y1:y2, x1:x2] = 255

            # Inflate points into small disks as a stand-in
            for p in frame_points:
                cx, cy = int(p.x), int(p.y)
                cv2.circle(mask, (cx, cy), radius=8, color=255, thickness=-1)

            # Merge any direct masks
            for m in frame_masks:
                mm = (m.mask.astype(bool)).astype(np.uint8) * 255
                mask = np.maximum(mask, mm)

            # If no per-frame prompt present, carry forward last mask (super naive temporal prior)
            if mask.sum() == 0 and i > 0:
                mask = accumulated_masks[i-1].copy()

            accumulated_masks[i] = mask

            # Save mask and overlay
            mask_name = fp.name.replace(fp.suffix, ".png")
            overlay = self._draw_overlay(img, mask, color=(0, 0, 255), alpha=0.4)

            cv2.imwrite(str(out_masks_dir / mask_name), mask)
            cv2.imwrite(str(out_overlays_dir / mask_name), overlay)

            pct = int(100.0 * (i+1) / total)
            progress_cb(pct, f"Processed frame {i+1}/{total}")

        # Note: Replace the above fallback with the true SAM2 propagation pipeline
        # using your installed SAM2 video predictor API.

        return PropagationResult(object_labels=obj_labels)
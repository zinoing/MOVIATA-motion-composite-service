from pathlib import Path
from PIL import Image
import numpy as np
import threading
import tempfile
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from concurrent.futures import ThreadPoolExecutor
from app.core.config import SAM2_CHECKPOINT_DIR

_CHECKPOINT = str(SAM2_CHECKPOINT_DIR / "sam2.1_hiera_small.pt")
_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_init_lock = threading.Lock()
_predictor: SAM2ImagePredictor | None = None
_infer_lock = threading.Lock()


def _get_predictor() -> SAM2ImagePredictor:
    global _predictor
    if _predictor is None:
        with _init_lock:
            if _predictor is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_sam2(_MODEL_CFG, _CHECKPOINT, device=device)
                _predictor = SAM2ImagePredictor(model)
    return _predictor


def _process_frame(frame_path: Path, points: list[dict] | None = None) -> dict:
    img = Image.open(frame_path).convert("RGB")
    rgb = np.array(img)
    h, w = rgb.shape[:2]

    person_coords: list[list[int]] = []
    object_coords: list[list[int]] = []

    if points:
        for p in points:
            px, py = int(p["x"] * w), int(p["y"] * h)
            if p.get("type") == "object":
                object_coords.append([px, py])
            else:
                person_coords.append([px, py])

    if not person_coords and not object_coords:
        person_coords = [[w // 2, h // 2]]

    person_mask: np.ndarray | None = None
    object_mask: np.ndarray | None = None

    with _infer_lock:
        predictor = _get_predictor()

        with torch.inference_mode():
            if person_coords:
                predictor.set_image(rgb)  # ✅ person 전 초기화
                all_coords = person_coords + object_coords
                all_labels = [1] * len(person_coords) + [0] * len(object_coords)
                masks, _, _ = predictor.predict(
                    point_coords=np.array(all_coords, dtype=np.float32),
                    point_labels=np.array(all_labels, dtype=np.int32),
                    multimask_output=True,
                )
                person_mask = masks[int(np.argmax([m.sum() for m in masks]))]

            if object_coords:
                predictor.set_image(rgb)  # ✅ object 전 초기화
                all_coords = object_coords + person_coords
                all_labels = [1] * len(object_coords) + [0] * len(person_coords)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(all_coords, dtype=np.float32),
                    point_labels=np.array(all_labels, dtype=np.int32),
                    multimask_output=True,
                )
                object_mask = masks[int(scores.argmax())]

    if person_mask is not None and object_mask is not None:
        object_mask = np.logical_and(object_mask.astype(bool), ~person_mask.astype(bool))

    original = np.array(Image.open(frame_path).convert("RGBA"))
    result: dict = {"frame_path": frame_path, "person": None, "object": None, "background": None}

    tmp = Path(tempfile.gettempdir())

    if person_mask is not None:
        alpha = person_mask.astype(np.uint8) * 255
        arr = original.copy()
        arr[:, :, 3] = alpha
        result["person"] = Image.fromarray(arr, "RGBA")

    if object_mask is not None:
        alpha = object_mask.astype(np.uint8) * 255
        arr = original.copy()
        arr[:, :, 3] = alpha
        result["object"] = Image.fromarray(arr, "RGBA")

    return result


def _frame_index_from_path(p: Path) -> int:
    try:
        return int(p.stem.replace("frame_", ""))
    except ValueError:
        return -1


def apply_masks(frames: list[Path], job_id: str, point_coords: list[dict] | None = None) -> list[dict]:  # noqa: ARG001
    coord_map: dict[int, list[dict]] = {}
    if point_coords:
        for c in point_coords:
            coord_map[c["frame_index"]] = c.get("points", [])

    def _process(frame_path: Path) -> dict:
        idx = _frame_index_from_path(frame_path)
        pts = coord_map.get(idx)
        return _process_frame(frame_path, pts)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_process, frames))
    return results
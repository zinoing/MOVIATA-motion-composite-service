import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from app.core.config import MAX_VIDEO_DURATION_SEC, TEMP_FRAMES_DIR


@dataclass
class ExtractedFrame:
    path: Path
    frame_index: int
    timestamp_sec: float


@dataclass
class ExtractionResult:
    frames: list[ExtractedFrame]
    output_dir: Path
    fps: float
    total_frames: int
    duration_sec: float
    frame_interval: int


def load_frames_by_indices(frame_indices: list[int], job_id: str) -> ExtractionResult:
    """Return ExtractionResult using already-extracted frames at specific indices."""
    output_dir = TEMP_FRAMES_DIR / job_id
    if not output_dir.exists():
        raise ValueError(f"No extracted frames directory found for job_id: {job_id}")

    frames: list[ExtractedFrame] = []
    for idx in sorted(frame_indices):
        path = output_dir / f"frame_{idx:06d}.png"
        if not path.exists():
            raise ValueError(f"Frame {idx} not found for job_id: {job_id}. Re-upload the video.")
        frames.append(ExtractedFrame(path=path, frame_index=idx, timestamp_sec=0.0))

    if not frames:
        raise ValueError(f"No valid frames found for job_id: {job_id}")

    return ExtractionResult(
        frames=frames,
        output_dir=output_dir,
        fps=30.0,
        total_frames=len(frames),
        duration_sec=0.0,
        frame_interval=1,
    )


def extract_single_image(image_path: str, job_id: str) -> ExtractionResult:
    """단일 이미지를 frame_000000.png 로 변환하여 1-프레임 ExtractionResult 를 반환한다."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image file: {image_path}") from exc

    output_dir = TEMP_FRAMES_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_path = output_dir / "frame_000000.png"
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(frame_path), bgr):
        raise IOError(f"Failed to write image frame to {frame_path}")

    return ExtractionResult(
        frames=[ExtractedFrame(path=frame_path, frame_index=0, timestamp_sec=0.0)],
        output_dir=output_dir,
        fps=0.0,
        total_frames=1,
        duration_sec=0.0,
        frame_interval=1,
    )


def extract_frames(video_path: str, frame_interval: int, job_id: str) -> ExtractionResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    if duration_sec > MAX_VIDEO_DURATION_SEC:
        cap.release()
        raise ValueError(
            f"Video duration {duration_sec:.1f}s exceeds the {MAX_VIDEO_DURATION_SEC}s limit"
        )

    output_dir = TEMP_FRAMES_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[ExtractedFrame] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            path = output_dir / f"frame_{frame_idx:06d}.png"
            success = cv2.imwrite(str(path), frame)
            if not success:
                cap.release()
                raise IOError(f"Failed to write frame {frame_idx} to {path}")
            extracted.append(
                ExtractedFrame(
                    path=path,
                    frame_index=frame_idx,
                    timestamp_sec=round(frame_idx / fps, 4),
                )
            )

        frame_idx += 1

    cap.release()

    return ExtractionResult(
        frames=extracted,
        output_dir=output_dir,
        fps=fps,
        total_frames=total_frames,
        duration_sec=round(duration_sec, 3),
        frame_interval=frame_interval,
    )

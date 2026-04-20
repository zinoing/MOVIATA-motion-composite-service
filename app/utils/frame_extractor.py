import cv2
from dataclasses import dataclass
from pathlib import Path

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

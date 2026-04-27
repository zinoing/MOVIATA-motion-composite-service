"""
In-process background pipeline (no Redis/Celery required).
Runs each job in a daemon thread and tracks state in memory.
"""

import threading
from pathlib import Path
from typing import Any

from app.utils.compositor import save_layers
from app.utils.masking import apply_masks
from app.utils.outliner import apply_outlines

_lock = threading.Lock()
_states: dict[str, dict[str, Any]] = {}


def get_state(job_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_states.get(job_id, {"status": "PENDING"}))


def run(
    job_id: str,
    frame_paths: list[str],
    person_color: str,
    background_color: str,
    outline_thickness: int,
    mode: str = "ghost",
    point_coords: list[dict] | None = None,
) -> None:
    thread = threading.Thread(
        target=_execute,
        args=(job_id, frame_paths, person_color, background_color, outline_thickness, mode, point_coords),
        daemon=True,
    )
    thread.start()


def _set(job_id: str, status: str, **kwargs: Any) -> None:
    with _lock:
        _states[job_id] = {"status": status, **kwargs}


def _execute(
    job_id: str,
    frame_paths: list[str],
    person_color: str,
    background_color: str,
    outline_thickness: int,
    mode: str,
    point_coords: list[dict] | None = None,
) -> None:
    try:
        _set(job_id, "PROGRESS", step="masking", progress=0)
        masked = apply_masks([Path(p) for p in frame_paths], job_id, point_coords)

        _set(job_id, "PROGRESS", step="outlining", progress=50)
        style = "halftone" if mode == "ghost" else "outline"
        frames_to_composite = apply_outlines(
            masked,
            person_color="#ffffff",    # ✅ person → 흰색
            background_color=background_color,
            thickness=outline_thickness,
            style=style,
            object_color="#FF5A1F",    # ✅ object → 주황색
        )

        _set(job_id, "PROGRESS", step="compositing", progress=75)
        layers = save_layers(frames_to_composite, job_id)

        _set(job_id, "SUCCESS", result={
            "job_id": job_id,
            "layers": layers,
        })

    except Exception as exc:
        _set(job_id, "FAILURE", error=str(exc))
import base64
import json
import re
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    OUTPUTS_DIR,
    TEMP_FRAMES_DIR,
)
from app.processing.pipeline import get_state, run as run_pipeline
from app.tasks.composite import process_video_task
from app.utils.frame_extractor import extract_frames, extract_single_image

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(default=60, ge=1, le=120),
    person_color: str = Form(default="#FF69B4"),
    background_color: str = Form(default="#FFFFFF"),
    outline_thickness: int = Form(default=2, ge=1, le=10),
    output_format: str = Form(default="png", pattern="^(png|mp4)$"),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: {MAX_UPLOAD_SIZE_MB}MB",
        )

    job_id = str(uuid.uuid4())
    video_path = TEMP_FRAMES_DIR / f"{job_id}{ext}"
    video_path.write_bytes(content)

    task = process_video_task.delay(
        job_id=job_id,
        video_path=str(video_path),
        frame_interval=frame_interval,
        person_color=person_color,
        background_color=background_color,
        outline_thickness=outline_thickness,
        output_format=output_format,
    )

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "celery_task_id": task.id, "status": "queued"},
    )


@router.post("/extract-frames")
async def extract_frames_endpoint(
    file: UploadFile = File(...),
    n: int = Form(default=5, ge=1, le=120, description="Extract every Nth frame"),
):
    ext = Path(file.filename).suffix.lower()
    is_image = ext in ALLOWED_IMAGE_EXTENSIONS
    if not is_image and ext not in ALLOWED_VIDEO_EXTENSIONS:
        all_allowed = sorted(ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(all_allowed)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: {MAX_UPLOAD_SIZE_MB}MB",
        )

    job_id = str(uuid.uuid4())
    file_path = TEMP_FRAMES_DIR / f"{job_id}{ext}"
    file_path.write_bytes(content)

    try:
        if is_image:
            result = extract_single_image(str(file_path), job_id)
        else:
            result = extract_frames(str(file_path), n, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except IOError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "job_id": job_id,
        "frame_interval": result.frame_interval,
        "fps": result.fps,
        "total_video_frames": result.total_frames,
        "duration_sec": result.duration_sec,
        "frames_extracted": len(result.frames),
        "output_dir": str(result.output_dir),
        "frames": [
            {
                "index": f.frame_index,
                "timestamp_sec": f.timestamp_sec,
                "path": str(f.path),
                "data": base64.b64encode(f.path.read_bytes()).decode(),
            }
            for f in result.frames
        ],
    }


@router.post("/process")
async def process_composite(
    job_id: str = Form(...),
    frame_paths: list[str] = Form(...),
    frame_data: str | None = Form(default=None),
    person_color: str = Form(default="#000000"),
    background_color: str = Form(default="#000000"),
    outline_thickness: int = Form(default=2, ge=1, le=10),
    mode: str = Form(default="ghost", pattern="^(ghost|outline)$"),
    point_coords: str | None = Form(default=None),
):
    if frame_data is not None:
        # RunPod: frames were extracted on a different worker — restore from base64 payload.
        try:
            data_list: list[str] = json.loads(frame_data)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid frame_data JSON")
        if len(data_list) != len(frame_paths):
            raise HTTPException(status_code=422, detail="frame_data length must match frame_paths")

        frame_dir = TEMP_FRAMES_DIR / job_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        resolved_paths: list[str] = []
        for path_str, b64_str in zip(frame_paths, data_list):
            frame_name = Path(path_str).name
            if not re.fullmatch(r"frame_\d{6}\.png", frame_name):
                raise HTTPException(status_code=422, detail=f"Invalid frame filename: {frame_name}")
            local_path = frame_dir / frame_name
            try:
                local_path.write_bytes(base64.b64decode(b64_str))
            except Exception:
                raise HTTPException(status_code=422, detail=f"Invalid base64 data for {frame_name}")
            resolved_paths.append(str(local_path))
        frame_paths = resolved_paths
    else:
        temp_root = str(TEMP_FRAMES_DIR.resolve())
        for p in frame_paths:
            try:
                resolved = Path(p).resolve()
                if not str(resolved).startswith(temp_root):
                    raise HTTPException(status_code=422, detail=f"Invalid frame path: {p}")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=422, detail=f"Invalid frame path: {p}")

    parsed_point_coords: list[dict] | None = None
    if point_coords:
        try:
            parsed_point_coords = json.loads(point_coords)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid point_coords JSON")

    run_pipeline(
        job_id=job_id,
        frame_paths=frame_paths,
        person_color=person_color,
        background_color=background_color,
        outline_thickness=outline_thickness,
        mode=mode,
        point_coords=parsed_point_coords,
    )

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "celery_task_id": job_id, "status": "queued"},
    )


@router.get("/layer/{job_id}/{index}")
async def get_layer(job_id: str, index: int):
    layer_path = OUTPUTS_DIR / f"{job_id}_layer_{index}.png"
    if not layer_path.exists():
        raise HTTPException(status_code=404, detail="Layer not found")
    return FileResponse(path=str(layer_path), media_type="image/png")


@router.get("/frame/{job_id}/{frame_index}")
async def get_frame(
    job_id: str,
    frame_index: int,
    w: int = Query(default=None, gt=0, le=3840),
):
    frame_path = TEMP_FRAMES_DIR / job_id / f"frame_{frame_index:06d}.png"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    if w is None:
        return FileResponse(path=str(frame_path), media_type="image/png")
    img = Image.open(frame_path).convert("RGB")
    if w < img.width:
        h = round(img.height * w / img.width)
        img = img.resize((w, h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/status/{task_id}")
async def get_job_status(task_id: str):
    state = get_state(task_id)

    response: dict = {"task_id": task_id, "status": state["status"]}

    if state["status"] == "SUCCESS":
        response["result"] = state.get("result")
    elif state["status"] == "FAILURE":
        response["error"] = state.get("error")
    elif state["status"] == "PROGRESS":
        response["progress"] = {"step": state.get("step"), "progress": state.get("progress", 0)}

    return response


@router.get("/result/{job_id}")
async def get_result(job_id: str, format: str = "png"):
    ext = "mp4" if format == "mp4" else "png"
    output_path = OUTPUTS_DIR / f"{job_id}.{ext}"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result not ready yet.")

    media_type = "video/mp4" if ext == "mp4" else "image/png"
    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=f"moviata-motion-{job_id[:8]}.{ext}",
    )

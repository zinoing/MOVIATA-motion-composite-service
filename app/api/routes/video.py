import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    TEMP_FRAMES_DIR,
)
from app.tasks.composite import process_video_task
from app.utils.frame_extractor import extract_frames

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(default=5, ge=1, le=30),
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

    try:
        result = extract_frames(str(video_path), n, job_id)
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
            }
            for f in result.frames
        ],
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    from app.tasks.composite import process_video_task
    from celery.result import AsyncResult

    result = AsyncResult(job_id, app=process_video_task.app)
    response = {"job_id": job_id, "status": result.state}

    if result.state == "SUCCESS":
        response["result"] = result.get()
    elif result.state == "FAILURE":
        response["error"] = str(result.info)
    elif result.state == "PROGRESS":
        response["progress"] = result.info

    return response

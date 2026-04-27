from app.tasks.celery_app import celery_app
from app.utils.compositor import composite_frames
from app.utils.frame_extractor import extract_frames
from app.utils.masking import apply_masks
from app.utils.outliner import apply_outlines


@celery_app.task(bind=True, name="tasks.process_video")
def process_video_task(
    self,
    job_id: str,
    video_path: str,
    frame_interval: int,
    person_color: str,
    background_color: str,
    outline_thickness: int,
    output_format: str,
    mode: str = "ghost",
):
    self.update_state(state="PROGRESS", meta={"step": "extracting_frames", "progress": 0})
    result = extract_frames(video_path, frame_interval, job_id)

    self.update_state(state="PROGRESS", meta={"step": "masking", "progress": 25})
    masked_frames = apply_masks([f.path for f in result.frames], job_id)

    self.update_state(state="PROGRESS", meta={"step": "outlining", "progress": 55})
    frames_to_composite = apply_outlines(masked_frames, person_color, background_color, outline_thickness)

    self.update_state(state="PROGRESS", meta={"step": "compositing", "progress": 75})
    output_path = composite_frames(frames_to_composite, job_id, output_format, mode=mode)

    self.update_state(state="PROGRESS", meta={"step": "done", "progress": 100})
    return {"job_id": job_id, "output_path": output_path, "format": output_format, "mode": mode}

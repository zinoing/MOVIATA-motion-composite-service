from celery import Celery
from app.core.config import REDIS_URL

celery_app = Celery(
    "motion_composite",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.composite"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)

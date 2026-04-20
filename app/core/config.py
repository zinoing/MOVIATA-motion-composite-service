from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TEMP_FRAMES_DIR = BASE_DIR / "temp_frames"
OUTPUTS_DIR = BASE_DIR / "outputs"

TEMP_FRAMES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "60"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TEMP_FRAMES_DIR = BASE_DIR / "temp_frames"
OUTPUTS_DIR = BASE_DIR / "outputs"

TEMP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

SAM2_CHECKPOINT_DIR = Path(os.getenv("SAM2_CHECKPOINT_DIR", str(BASE_DIR / "checkpoints" / "sam2")))

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "60"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

R2_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_TEMP = os.getenv("R2_BUCKET_TEMP", "moviata-temp")
R2_ENDPOINT_URL = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""

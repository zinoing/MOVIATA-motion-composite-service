"""RunPod Serverless handler.

Boots FastAPI + Celery as subprocesses on pod start,
waits for readiness, then proxies every RunPod job to the local FastAPI server.

Expected job["input"] shape:
{
  "endpoint": "/api/video/extract-frames",  // FastAPI path
  "method":   "POST",                        // optional, default "GET"
  "body": {                                  // optional: form fields
    "job_id": "...",
    "frame_paths": ["path1", "path2"]        // list values are expanded
  },
  "files": {                                 // optional: multipart file upload
    "file": {
      "filename":     "video.mp4",
      "content_type": "video/mp4",
      "data":         "<base64-encoded bytes>"
    }
  }
}

Image responses (image/*) are returned as:
{ "content_type": "image/png", "encoding": "base64", "data": "<base64>" }

All other responses are returned as the JSON body from FastAPI.
"""

import base64
import io
import os
import pathlib
import subprocess
import sys
import time

import requests
import runpod

# ── Constants ─────────────────────────────────────────────────────────────────

FASTAPI_BASE = "http://localhost:8000"
HEALTH_URL   = f"{FASTAPI_BASE}/health"
BOOT_TIMEOUT = 120  # seconds to wait for FastAPI readiness

# ── Checkpoint download (runs once at pod startup) ─────────────────────────────

def _download_checkpoint() -> None:
    r2_url = os.environ.get("R2_CHECKPOINT_URL", "")
    checkpoint_dir = pathlib.Path(os.environ.get("SAM2_CHECKPOINT_DIR", "checkpoints/sam2"))
    checkpoint_path = checkpoint_dir / "sam2.1_hiera_small.pt"
    if checkpoint_path.exists():
        print("[handler] Checkpoint already exists, skipping download.", flush=True)
        return
    if not r2_url:
        raise RuntimeError("R2_CHECKPOINT_URL is not set")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("[handler] Downloading SAM2 checkpoint from R2...", flush=True)
    subprocess.run(["curl", "-L", "-o", str(checkpoint_path), r2_url], check=True)
    print("[handler] Checkpoint downloaded.", flush=True)


# ── Service bootstrap (runs once at pod startup) ───────────────────────────────

def _start_services() -> None:
    """Start Celery worker and uvicorn as background processes."""
    subprocess.Popen(
        [
            sys.executable, "-m", "celery",
            "-A", "app.tasks.celery_app",
            "worker",
            "--loglevel=warning",
            "--concurrency=2",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_ready() -> None:
    """Poll /health until FastAPI responds or timeout is reached."""
    deadline = time.time() + BOOT_TIMEOUT
    while time.time() < deadline:
        try:
            if requests.get(HEALTH_URL, timeout=2).ok:
                print("[handler] FastAPI is ready", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"FastAPI did not become ready within {BOOT_TIMEOUT}s")


_download_checkpoint()
_start_services()
_wait_ready()

# ── Request helper ─────────────────────────────────────────────────────────────

def _build_form_data(body: dict) -> list[tuple[str, str]]:
    """Expand list values so FastAPI receives repeated form keys."""
    pairs: list[tuple[str, str]] = []
    for key, val in body.items():
        if isinstance(val, list):
            for item in val:
                pairs.append((key, str(item)))
        else:
            pairs.append((key, str(val)))
    return pairs


def _decode_files(files_b64: dict) -> dict:
    """Decode base64 file payloads into (filename, BytesIO, content_type) tuples."""
    return {
        field: (
            meta["filename"],
            io.BytesIO(base64.b64decode(meta["data"])),
            meta.get("content_type", "application/octet-stream"),
        )
        for field, meta in files_b64.items()
    }

# ── Handler ────────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp       = job.get("input") or {}
    endpoint  = inp.get("endpoint", "/health")
    method    = inp.get("method", "GET").upper()
    body      = inp.get("body") or {}
    files_b64 = inp.get("files") or {}

    url       = f"{FASTAPI_BASE}{endpoint}"
    form_data = _build_form_data(body)

    try:
        if method == "GET":
            resp = requests.get(url, params=dict(form_data), timeout=300)

        elif files_b64:
            resp = requests.post(
                url,
                data=form_data,
                files=_decode_files(files_b64),
                timeout=300,
            )

        else:
            resp = requests.post(url, data=form_data, timeout=300)

    except requests.RequestException as exc:
        return {"error": str(exc)}

    # Binary image → base64 payload
    content_type = resp.headers.get("content-type", "")
    if resp.status_code == 200 and content_type.startswith("image/"):
        return {
            "content_type": content_type,
            "encoding":     "base64",
            "data":         base64.b64encode(resp.content).decode(),
        }

    # JSON (or fallback text)
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}

    if not resp.ok:
        return {"error": payload}

    return payload


runpod.serverless.start({"handler": handler})

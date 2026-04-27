from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.video import router as video_router
from app.core.config import CORS_ORIGINS

app = FastAPI(
    title="MOVIATA Motion Composite Service",
    description="Stroboscopic ghost-frame motion composite pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.video import router as video_router

app = FastAPI(
    title="MOVIATA Motion Composite Service",
    description="Stroboscopic ghost-frame motion composite pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}

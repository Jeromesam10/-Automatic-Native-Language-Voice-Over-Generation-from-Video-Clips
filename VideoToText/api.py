import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import BASE_DIR, FRAME_INTERVAL
from agent import run

app = FastAPI(title="Video To Text API")


class VideoRequest(BaseModel):
    video_path: str
    frame_interval: int = FRAME_INTERVAL


@app.get("/")
def health():
    return {"status": "Running", "service": "video_to_text"}


@app.post("/video")
def video(request: VideoRequest):

    # Relative paths are resolved against the VideoToText folder,
    # e.g. "videos/Video-525.mp4".
    video_path = request.video_path
    if not os.path.isabs(video_path):
        video_path = os.path.join(BASE_DIR, video_path)

    if request.frame_interval < 1:
        raise HTTPException(status_code=422, detail="frame_interval must be >= 1")

    try:
        caption = run(video_path, request.frame_interval)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "video_path": video_path,
        "caption": caption.strip()
    }

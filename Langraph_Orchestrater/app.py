"""Orchestrator API (port 8080). Two flows:

  POST /run  narration: describe what the video shows -> translate ->
             cloned voice-over -> narrated video
  POST /dub  dubbing:   translate what people say -> speaker's cloned voice
             at the original timing -> lip sync -> dubbed video
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER_REFERENCE,
    DUBBING_API_URL,
    DUBBING_TIMEOUT,
)
from graph import graph
from nodes import ServiceError, _post
from paths import InputError, resolve_speaker_reference, resolve_video

app = FastAPI(title="LangGraph Orchestrator API")


class RequestData(BaseModel):
    input: str  # video path: absolute, or relative to the project root / VideoToText/
    language: str = DEFAULT_LANGUAGE
    speaker_reference: str = DEFAULT_SPEAKER_REFERENCE  # relative = Voice_Clone/


class DubRequest(BaseModel):
    video_path: str
    target_language: str = DEFAULT_LANGUAGE
    source_language: str | None = None
    tts_backend: str | None = None        # "clone" (default) or "edge"
    speaker_reference: str | None = None  # default: voice taken from the video
    lip_sync: bool | None = None          # re-render the speaker's mouth (default on)


@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "LangGraph Orchestrator API"
    }


@app.post("/run")
def run_graph(data: RequestData):
    """Narration mode."""
    try:
        initial_state = {
            "input": resolve_video(data.input),
            "language": data.language,
            "speaker_reference": resolve_speaker_reference(data.speaker_reference),
        }
    except InputError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        result = graph.invoke(initial_state)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    return result["final_output"]


@app.post("/dub")
def dub(data: DubRequest):
    """Dubbing mode."""
    payload = data.model_dump()
    try:
        payload["video_path"] = resolve_video(data.video_path)
        if data.speaker_reference:
            payload["speaker_reference"] = resolve_speaker_reference(data.speaker_reference)
    except InputError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        return _post("Dubbing API", DUBBING_API_URL, payload, DUBBING_TIMEOUT)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import dub_video
from synth import SynthesisError
from translate import TranslationError
from pipeline import LipSyncFailed

app = FastAPI(title="Video Dubbing API")


class DubRequest(BaseModel):
    video_path: str
    target_language: str                  # e.g. "hi", "ta", "fr"
    source_language: str | None = None    # auto-detected when omitted
    tts_backend: str | None = None        # "clone" (default) or "edge"
    speaker_reference: str | None = None  # optional; default = voice taken from the video
    separate_vocals: bool | None = None   # keep music/effects under the dub
    lip_sync: bool | None = None          # re-render the speaker's mouth (default on)


class NarrateRequest(BaseModel):
    video_path: str
    voice_path: str                 # narration audio (e.g. from Voice_Clone)
    original_volume: float = 0.25   # keep the original sound, lowered


@app.post("/narrate")
def narrate(request: NarrateRequest):
    """Narration mode, last step: put the voice-over onto the video."""
    import uuid
    from pathlib import Path

    import media
    from config import OUTPUT_DIR

    video, voice = Path(request.video_path), Path(request.voice_path)
    for p in (video, voice):
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {p}")
    out_dir = OUTPUT_DIR / f"narration_{uuid.uuid4().hex[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out, duration = media.narrate(video, voice, out_dir / f"{video.stem}.narrated.mp4",
                                      request.original_volume)
        voice_seconds = media.audio_duration_any(voice)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result = {"narrated_video": str(out), "video_seconds": round(duration, 2),
              "narration_seconds": round(voice_seconds, 2), "warnings": []}
    if voice_seconds > duration + 0.05:
        result["warnings"].append(
            f"Narration is {voice_seconds - duration:.1f}s longer than the video; the end was cut.")
    return result


@app.get("/")
def health():
    return {"status": "Running", "service": "dubbing"}


@app.post("/dub")
def dub(request: DubRequest):
    try:
        return dub_video(
            request.video_path,
            request.target_language,
            source_language=request.source_language,
            tts_backend=request.tts_backend,
            speaker_reference=request.speaker_reference,
            separate=request.separate_vocals,
            lip_sync=request.lip_sync,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (SynthesisError, TranslationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except LipSyncFailed as exc:
        raise HTTPException(status_code=500, detail=f"Lip sync failed: {exc}")

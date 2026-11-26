from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import run

app = FastAPI(title="Voice Clone API")


class VoiceRequest(BaseModel):
    translated_text: str
    speaker_reference: str = "voices/reference.wav"
    language: str | None = None


@app.get("/")
def health():
    return {"status": "Running", "service": "voice_clone"}


@app.post("/voice")
def voice(request: VoiceRequest):
    try:
        return run(
            translated_text=request.translated_text,
            speaker_reference=request.speaker_reference,
            language=request.language,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

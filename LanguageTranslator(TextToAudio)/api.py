from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import TranslationError, run

app = FastAPI(title="Translator API")


class TranslatorRequest(BaseModel):
    text: str
    language: str                 # target language, e.g. "ta"
    source_language: str = "en"   # VideoToText captions are English
    make_audio: bool = True       # gTTS preview MP3 (optional)


@app.get("/")
def health():
    return {"status": "Running", "service": "translator"}


@app.post("/translate")
def translate(request: TranslatorRequest):
    try:
        return run(request.text, request.language, request.source_language,
                   make_audio=request.make_audio)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except TranslationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

from fastapi import FastAPI
from pydantic import BaseModel

from agent import run

app = FastAPI()


class VoiceRequest(BaseModel):
    translated_text: str
    speaker_reference: str
    language: str


@app.post("/voice")
def voice(request: VoiceRequest):

    result = run(
        translated_text=request.translated_text,
        speaker_reference=request.speaker_reference,
        language=request.language
    )

    return result
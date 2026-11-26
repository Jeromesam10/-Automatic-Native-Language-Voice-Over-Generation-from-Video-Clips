"""Speech synthesis for each translated segment.

Backends:
  clone - Voice_Clone service (XTTS for its 17 languages; Edge TTS + FreeVC
          voice conversion for Indian languages). Sounds like the speaker.
  edge  - stock Microsoft Edge neural voice, no cloning. Fast; for testing.
"""
import asyncio
from pathlib import Path

import requests

from config import VOICE_API_URL, VOICE_TIMEOUT

EDGE_VOICES = {
    "en": "en-US-GuyNeural", "hi": "hi-IN-MadhurNeural", "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural", "ml": "ml-IN-MidhunNeural", "kn": "kn-IN-GaganNeural",
    "bn": "bn-IN-BashkarNeural", "mr": "mr-IN-ManoharNeural", "gu": "gu-IN-NiranjanNeural",
    "pa": "pa-IN-OjasNeural", "ur": "ur-IN-SalmanNeural", "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural", "de": "de-DE-ConradNeural", "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural", "ru": "ru-RU-DmitryNeural", "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural", "zh": "zh-CN-YunxiNeural", "ar": "ar-SA-HamedNeural",
    "tr": "tr-TR-AhmetNeural", "nl": "nl-NL-MaartenNeural", "pl": "pl-PL-MarekNeural",
}

# Voice_Clone/XTTS language codes where they differ from ISO codes.
_CLONE_CODES = {"zh": "zh-cn", "zh-cn": "zh-cn"}


class SynthesisError(Exception):
    pass


def synthesize(text, language, out_path, backend, speaker_reference=None, voice=None):
    """Write speech for `text` and return the path of the produced audio file."""
    if backend == "edge":
        return _edge(text, language, Path(out_path).with_suffix(".mp3"), voice)
    if backend == "clone":
        return _clone(text, language, speaker_reference)
    raise ValueError(f"Unknown TTS backend '{backend}' (use 'clone' or 'edge')")


def _edge(text, language, out_path, voice=None):
    import edge_tts

    voice = voice or EDGE_VOICES.get(language.lower().split("-")[0])
    if not voice:
        raise SynthesisError(f"No Edge voice configured for language '{language}'")

    async def _save():
        await edge_tts.Communicate(text, voice).save(str(out_path))

    asyncio.run(_save())
    return Path(out_path)


def _clone(text, language, speaker_reference):
    if not speaker_reference:
        raise SynthesisError("Voice cloning needs a speaker reference")
    lang = _CLONE_CODES.get(language.lower(), language.lower())
    try:
        response = requests.post(
            VOICE_API_URL,
            json={
                "translated_text": text,
                "speaker_reference": str(Path(speaker_reference).resolve()),
                "language": lang,
            },
            timeout=VOICE_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SynthesisError(f"Voice Clone API unreachable at {VOICE_API_URL}: {exc}") from exc
    if response.status_code != 200:
        raise SynthesisError(f"Voice Clone API error {response.status_code}: {response.text}")
    return Path(response.json()["voice_path"])

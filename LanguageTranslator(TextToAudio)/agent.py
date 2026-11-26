"""Translate text and (optionally) make a quick preview MP3 of it.

Translation reuses Dubbing/translate.py so both flows behave the same:
TRANSLATOR=local (default) runs facebook/m2m100_418M offline; TRANSLATOR=google
uses the free Google endpoint, which is rate-limited and can be blocked.

The preview MP3 uses gTTS (Google). It is not used downstream — the Voice
Clone service makes the real voice-over — so if it fails, audio_path is None.
"""
import sys
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

sys.path.insert(0, str(BASE_DIR.parent / "Dubbing"))
from translate import TranslationError, translate_texts  # noqa: E402


def run(text, target_language, source_language="en", output_file=None, make_audio=True):
    """
    Returns:
        {"translated_text": "...", "audio_path": "<abs path to mp3>" or None,
         "audio_error": None or "<why no preview audio>"}
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("No text to translate")

    translated = translate_texts([text], target_language, source_language)[0]

    audio_path, audio_error = None, None
    if make_audio:
        if output_file is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"translation_{uuid.uuid4().hex}.mp3"
        try:
            from gtts import gTTS
            gTTS(text=translated, lang=target_language).save(str(output_file))
            audio_path = str(Path(output_file).resolve())
        except Exception as exc:          # preview only; never fail the request
            audio_error = f"Preview audio not created: {exc}"

    return {
        "translated_text": translated,
        "audio_path": audio_path,
        "audio_error": audio_error,
    }


__all__ = ["run", "TranslationError"]

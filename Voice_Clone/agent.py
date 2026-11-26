import uuid
from pathlib import Path

from convert import text_to_speech, subtitles_to_speech

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def _resolve(path):
    """Resolve a path relative to the Voice_Clone folder, not the caller's cwd."""
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def _new_output_path():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"voice_{uuid.uuid4().hex}.wav"


def run(
    translated_text=None,
    speaker_reference="voices/reference.wav",
    language=None,
    subtitle_file=None,
):
    """Synthesize `translated_text` (or the text in `subtitle_file`) in the
    cloned voice of `speaker_reference`.

    Returns:
        {"voice_path": "<absolute path to wav>"}
    """
    speaker_wav = _resolve(speaker_reference)
    output_path = _new_output_path()

    if translated_text is not None:
        output = text_to_speech(
            text=translated_text,
            output_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
        )
    elif subtitle_file is not None:
        output = subtitles_to_speech(
            subtitles_path=_resolve(subtitle_file),
            output_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
        )
    else:
        raise ValueError("Provide either translated_text or subtitle_file")

    return {"voice_path": str(output)}

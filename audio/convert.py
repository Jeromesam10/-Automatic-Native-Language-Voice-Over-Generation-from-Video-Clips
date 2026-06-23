from __future__ import annotations

from pathlib import Path

import torch
from TTS.api import TTS

_AUDIO_DIR = Path(__file__).resolve().parent
VOICES_DIR = _AUDIO_DIR / "voices"
SUBTITLES_FILE = _AUDIO_DIR / "subtitles.txt"
OUTPUT_AUDIO_FILE = _AUDIO_DIR / "voiceover.wav"

_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts: TTS | None = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS(_MODEL_NAME).to(_get_device())
    return _tts


def _normalize_speaker_wav(
    speaker_wav: Path | str | list[Path | str],
) -> list[str]:
    if isinstance(speaker_wav, (str, Path)):
        paths = [Path(speaker_wav)]
    else:
        paths = [Path(path) for path in speaker_wav]

    resolved = [str(path.resolve()) for path in paths]
    missing = [path for path in resolved if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Reference voice file not found: "
            + ", ".join(missing)
            + ". Record 6–30 seconds of clean speech and save it under audio/voices/."
        )
    return resolved


def subtitles_to_speech(
    subtitles_path: Path | str = SUBTITLES_FILE,
    output_path: Path | str = OUTPUT_AUDIO_FILE,
    speaker_wav: Path | str | list[Path | str] | None = None,
    speaker_id: str | None = None,
    language: str = "en",
) -> Path:

    subtitles_path = Path(subtitles_path)
    output_path = Path(output_path)

    if speaker_wav is None:
        speaker_wav = VOICES_DIR / "reference.wav"

    text = subtitles_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text found in {subtitles_path}")

    speaker_files = _normalize_speaker_wav(speaker_wav)
    tts = _get_tts()

    kwargs = {
        "text": text,
        "speaker_wav": speaker_files if len(speaker_files) > 1 else speaker_files[0],
        "language": language,
        "file_path": str(output_path),
    }
    if speaker_id:
        kwargs["speaker"] = speaker_id

    tts.tts_to_file(**kwargs)
    return output_path


if __name__ == "__main__":
    import sys

    speaker_arg = sys.argv[1] if len(sys.argv) > 1 else None
    speaker = speaker_arg or VOICES_DIR / "reference.wav"
    result = subtitles_to_speech(speaker_wav=speaker)
    print(f"Saved cloned speech audio to {result}")

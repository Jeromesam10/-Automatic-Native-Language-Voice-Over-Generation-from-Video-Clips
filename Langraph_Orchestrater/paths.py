"""Input validation shared by both flows, so bad input fails in milliseconds
instead of after minutes of model work."""
from pathlib import Path

from config import PROJECT_ROOT, VIDEO_TO_TEXT_DIR, VOICE_CLONE_DIR


class InputError(ValueError):
    pass


def resolve_video(path):
    """Absolute path of the input video. Relative paths are tried against the
    current folder, the project root, then VideoToText/ (old default)."""
    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, PROJECT_ROOT / p, VIDEO_TO_TEXT_DIR / p]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    raise InputError(f"Video not found: {path}")


def resolve_speaker_reference(path):
    """Absolute path of the narration voice sample (relative = Voice_Clone/)."""
    p = Path(path)
    full = p if p.is_absolute() else VOICE_CLONE_DIR / p
    if not full.is_file():
        raise InputError(
            f"Speaker reference not found: {full}. Narration mode clones the voice in "
            "this recording: add 6-30 s of clean speech as Voice_Clone/voices/reference.wav "
            "or pass speaker_reference.")
    return str(full.resolve())

"""Build the voice-cloning reference from the speaker's own speech in the
video, so the dub sounds like the original speaker."""
import numpy as np
import soundfile as sf

from config import REFERENCE_MAX_SECONDS, REFERENCE_MIN_SECONDS

_REFERENCE_RATE = 24000


def build_reference(vocals_path, segments, out_path):
    """Concatenate the longest speech segments (from the separated vocals)
    until ~REFERENCE_MAX_SECONDS. Returns (path, seconds)."""
    import media

    audio = media.load_clip(vocals_path, _REFERENCE_RATE)  # mono float32

    pad = 0.05
    ranked = sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True)
    pieces, total = [], 0.0
    silence = np.zeros(int(_REFERENCE_RATE * 0.2), dtype="float32")

    for seg in ranked:
        start = max(0, int((seg["start"] - pad) * _REFERENCE_RATE))
        end = min(len(audio), int((seg["end"] + pad) * _REFERENCE_RATE))
        chunk = audio[start:end]
        if len(chunk) < _REFERENCE_RATE * 0.8:      # skip tiny fragments
            continue
        room = REFERENCE_MAX_SECONDS - total
        chunk = chunk[: int(room * _REFERENCE_RATE)]
        pieces += [chunk, silence]
        total += len(chunk) / _REFERENCE_RATE
        if total >= REFERENCE_MAX_SECONDS - 0.5:
            break

    if total < REFERENCE_MIN_SECONDS:
        raise ValueError(
            f"Only {total:.1f}s of clear speech found; voice cloning needs at "
            f"least {REFERENCE_MIN_SECONDS:.0f}s. Use a longer clip or "
            "pass a speaker_reference recording."
        )

    ref = np.concatenate(pieces)
    peak = float(np.max(np.abs(ref))) or 1.0
    ref = ref / peak * 0.9
    sf.write(str(out_path), ref, _REFERENCE_RATE, subtype="PCM_16")
    return out_path, total

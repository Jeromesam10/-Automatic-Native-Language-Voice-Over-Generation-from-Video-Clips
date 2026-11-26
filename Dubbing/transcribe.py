"""Speech recognition with timestamps (faster-whisper)."""
from config import (
    MERGE_GAP_SECONDS,
    MERGE_MAX_SECONDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                              compute_type=WHISPER_COMPUTE_TYPE)
    return _model


def transcribe(audio_path, language=None):
    """Return (segments, detected_language).
    segments: [{"start": float, "end": float, "text": str}, ...]"""
    import media

    # Decode with ffmpeg ourselves (16 kHz mono float32, what Whisper expects)
    # instead of letting faster-whisper use PyAV, whose API changes between
    # versions.
    audio = media.load_clip(audio_path, 16000)

    segments, info = _get_model().transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    result = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            result.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return merge_segments(result), info.language


def merge_segments(segments, max_gap=MERGE_GAP_SECONDS, max_len=MERGE_MAX_SECONDS):
    """Join fragments that follow each other closely, so each TTS call gets a
    whole phrase (short fragments sound choppy when synthesized alone)."""
    merged = []
    for seg in segments:
        if merged:
            last = merged[-1]
            gap = seg["start"] - last["end"]
            if gap <= max_gap and seg["end"] - last["start"] <= max_len:
                last["end"] = seg["end"]
                last["text"] = f'{last["text"]} {seg["text"]}'
                continue
        merged.append(dict(seg))
    return merged

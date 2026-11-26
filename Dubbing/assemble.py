"""Fit each dubbed line into its original time slot and mix the new voice
track with the background."""
from pathlib import Path

import numpy as np
import soundfile as sf

import media
from config import BACKGROUND_GAIN, MAX_SPEEDUP, SAMPLE_RATE, VOICE_GAIN

_FADE_SECONDS = 0.08


def trim_silence(audio, sample_rate, threshold_db=-40.0, keep_seconds=0.05):
    """Cut leading/trailing audio quieter than `threshold_db` below the peak."""
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio[:0]
    window = max(1, int(0.02 * sample_rate))
    usable = len(audio) // window * window
    if usable == 0:
        return audio
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, window) ** 2, axis=1))
    loud = np.where(rms > peak * 10 ** (threshold_db / 20))[0]
    if len(loud) == 0:
        return audio
    keep = int(keep_seconds * sample_rate)
    start = max(0, loud[0] * window - keep)
    end = min(len(audio), (loud[-1] + 1) * window + keep)
    return audio[start:end]


def fit_segments(segments, total_duration, sample_rate=SAMPLE_RATE):
    """segments: dicts with start, end and `audio_file` (synthesized speech).
    Returns (mono voice track, per-segment timing report)."""
    track = np.zeros(int(total_duration * sample_rate) + sample_rate, dtype="float32")
    report = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        # A line may run into the silence before the next one, not over it.
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else total_duration
        slot = max(0.3, next_start - start)

        # TTS engines (XTTS especially) pad output with silence; measure and
        # fit only the speech itself.
        clip = trim_silence(media.load_clip(seg["audio_file"], sample_rate), sample_rate)
        raw_len = len(clip) / sample_rate
        tempo = 1.0
        if raw_len > slot:
            tempo = min(raw_len / slot, MAX_SPEEDUP)
            trimmed_path = Path(seg["audio_file"]).with_suffix(".trim.wav")
            media.write_wav(trimmed_path, clip, sample_rate)
            clip = media.load_clip(trimmed_path, sample_rate, tempo=tempo)
            trimmed_path.unlink(missing_ok=True)

        max_samples = int(slot * sample_rate)
        overflow = len(clip) / sample_rate - slot
        if len(clip) > max_samples:
            clip = clip[:max_samples].copy()
            fade = min(len(clip), int(_FADE_SECONDS * sample_rate))
            clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")

        offset = int(start * sample_rate)
        end = min(len(track), offset + len(clip))
        track[offset:end] += clip[: end - offset]

        report.append({
            "start": round(start, 2),
            "slot_seconds": round(slot, 2),
            "speech_seconds": round(raw_len, 2),
            "speedup": round(tempo, 2),
            "trimmed_seconds": round(max(0.0, overflow), 2),
            "placed_end": round(start + len(clip) / sample_rate, 3),
        })

    return track[: int(total_duration * sample_rate)], report


def mix(voice_track, background_path, out_path, sample_rate=SAMPLE_RATE):
    """Mix mono voice with a stereo background (or none) into a stereo WAV."""
    voice = voice_track * VOICE_GAIN
    stereo = np.stack([voice, voice], axis=1)

    if background_path:
        bg, sr = sf.read(str(background_path), dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise ValueError(f"Background must be {sample_rate} Hz, got {sr}")
        if bg.shape[1] == 1:
            bg = np.repeat(bg, 2, axis=1)
        n = min(len(bg), len(stereo))
        stereo = stereo[:n] + bg[:n, :2] * BACKGROUND_GAIN

    peak = float(np.max(np.abs(stereo))) if len(stereo) else 0.0
    if peak > 0.98:
        stereo = stereo / peak * 0.98
    return media.write_wav(out_path, stereo, sample_rate)

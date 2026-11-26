"""Video dubbing pipeline:

  video -> extract audio -> separate voice / background (Demucs)
        -> speech-to-text with timestamps (faster-whisper)
        -> translate each line
        -> clone the speaker's voice and speak each translated line
        -> fit each line into its original time slot
        -> mix with the background -> mux into the original video (+ subtitles)
"""
import json
import time
import uuid
from pathlib import Path

import assemble
import media
import separate as separation
import synth
import transcribe
import translate
from config import LIP_SYNC, LIPSYNC_DIR, OUTPUT_DIR, SAMPLE_RATE, SEPARATE_VOCALS, TTS_BACKEND


class LipSyncFailed(Exception):
    pass


def _free_stage_models():
    """Drop Demucs / Whisper / translation models before lip sync, which needs
    ~4 GB on its own (matters on machines with 8 GB of RAM)."""
    import gc
    separation._model = None
    transcribe._model = None
    translate._local = None
    gc.collect()


def _run_lip_sync(video_path, voice_wav, timing, job_dir, log):
    """Returns (video file to mux, lip-sync report)."""
    import sys
    if str(LIPSYNC_DIR) not in sys.path:
        sys.path.insert(0, str(LIPSYNC_DIR))
    import lipsync

    # Re-render only while dubbed speech is playing (+0.15 s either side).
    windows = [(max(0.0, t["start"] - 0.15), t["placed_end"] + 0.15) for t in timing]
    out = job_dir / "lipsync_video.mp4"
    try:
        report = lipsync.lip_sync(video_path, voice_wav, out, windows,
                                  ffmpeg=media.ffmpeg_exe(), log=log)
    except lipsync.InsufficientMemory as exc:
        # Not fatal: deliver the dub with the original picture.
        return video_path, {"applied": False, "reason": f"Lip sync skipped: {exc}"}
    except lipsync.LipSyncError as exc:
        raise LipSyncFailed(str(exc)) from exc
    return (out if report["applied"] else video_path), report


def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, key, path):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}", seg[key], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def dub_video(
    video_path,
    target_language,
    source_language=None,
    tts_backend=None,
    speaker_reference=None,
    separate=None,
    job_dir=None,
    lip_sync=None,
):
    """Dub `video_path` into `target_language`. Returns a dict describing the
    outputs (dubbed video, subtitles, transcript, timing report)."""
    t0 = time.time()
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    tts_backend = tts_backend or TTS_BACKEND
    separate = SEPARATE_VOCALS if separate is None else separate
    lip_sync = LIP_SYNC if lip_sync is None else lip_sync
    job_dir = Path(job_dir or OUTPUT_DIR / f"job_{uuid.uuid4().hex[:12]}")
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    # 1. Audio
    duration, has_audio = media.probe(video_path)
    if not has_audio:
        raise ValueError("The video has no audio track, so there is no speech to dub.")
    log(f"Video {duration:.1f}s - extracting audio")
    original_wav = media.extract_audio(video_path, job_dir / "original.wav", SAMPLE_RATE)

    # 2. Separate voice from background
    if separate:
        log("Separating voice from background (Demucs)")
        vocals_wav, background_wav = separation.separate_vocals(
            original_wav, job_dir / "vocals.wav", job_dir / "background.wav")
    else:
        log("Separation disabled - original audio is dropped, only the dub is kept")
        vocals_wav, background_wav = original_wav, None

    # 3. Transcribe
    log("Transcribing speech")
    segments, detected = transcribe.transcribe(vocals_wav, language=source_language)
    source_language = source_language or detected
    if not segments:
        raise ValueError("No speech was detected in the video.")
    log(f"{len(segments)} segments, source language '{source_language}'")

    # 4. Translate
    log(f"Translating {source_language} -> {target_language}")
    translated = translate.translate_texts(
        [s["text"] for s in segments], target_language, source_language)
    for seg, text in zip(segments, translated):
        seg["translated"] = text

    # 5. Voice reference from the speaker's own voice
    reference_seconds = None
    warnings = []
    if tts_backend == "clone" and not speaker_reference:
        import reference
        from config import REFERENCE_RECOMMENDED_SECONDS
        speaker_reference, reference_seconds = reference.build_reference(
            vocals_wav, segments, job_dir / "speaker_reference.wav")
        log(f"Built {reference_seconds:.1f}s speaker reference from the video")
        if reference_seconds < REFERENCE_RECOMMENDED_SECONDS:
            warnings.append(
                f"Only {reference_seconds:.1f}s of the speaker's voice was available; "
                f"cloning is more faithful with {REFERENCE_RECOMMENDED_SECONDS:.0f}s or more.")

    # 6. Synthesize each line
    for i, seg in enumerate(segments):
        log(f"Synthesizing {i + 1}/{len(segments)} ({tts_backend})")
        seg["audio_file"] = str(synth.synthesize(
            seg["translated"], target_language,
            job_dir / "segments" / f"seg_{i:04d}.wav",
            backend=tts_backend, speaker_reference=speaker_reference))

    # 7. Fit timing and mix
    log("Fitting lines to original timing and mixing")
    voice_track, timing = assemble.fit_segments(segments, duration)
    voice_wav = media.write_wav(job_dir / "dubbed_voice.wav", voice_track, SAMPLE_RATE)
    mixed_wav = assemble.mix(voice_track, background_wav, job_dir / "dubbed_audio.wav")

    # 8. Lip sync the speaker's mouth to the new voice
    picture = video_path
    lip_sync_report = {"applied": False, "reason": "Lip sync disabled"}
    if lip_sync:
        log("Lip sync (MuseTalk 1.5)")
        _free_stage_models()
        picture, lip_sync_report = _run_lip_sync(video_path, voice_wav, timing, job_dir, log)
        if not lip_sync_report["applied"]:
            log(lip_sync_report["reason"])
            warnings.append(lip_sync_report["reason"])

    # 9. Subtitles + final video
    srt_target = write_srt(segments, "translated", job_dir / f"subtitles.{target_language}.srt")
    srt_source = write_srt(segments, "text", job_dir / f"subtitles.{source_language}.srt")
    out_video = media.mux(picture, mixed_wav,
                          job_dir / f"{video_path.stem}.{target_language}.mp4",
                          subtitles_path=srt_target, subtitle_lang=target_language)
    log(f"Done -> {out_video}")

    result = {
        "dubbed_video": str(out_video),
        "dubbed_audio": str(mixed_wav),
        "subtitles": str(srt_target),
        "source_subtitles": str(srt_source),
        "source_language": source_language,
        "target_language": target_language,
        "tts_backend": tts_backend,
        "speaker_reference": str(speaker_reference) if speaker_reference else None,
        "reference_seconds": reference_seconds,
        "background_kept": background_wav is not None,
        "lip_sync": lip_sync_report,
        "duration_seconds": round(duration, 2),
        "processing_seconds": round(time.time() - t0, 1),
        "warnings": warnings,
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2),
             "text": s["text"], "translated": s["translated"], **timing[i]}
            for i, s in enumerate(segments)
        ],
    }
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return result

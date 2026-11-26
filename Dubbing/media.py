"""ffmpeg helpers. Uses the ffmpeg binary bundled with imageio-ffmpeg, so no
system-wide ffmpeg install is needed (a system ffmpeg on PATH also works)."""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg_exe():
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args):
    cmd = [ffmpeg_exe(), "-hide_banner", "-y", *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}):\n{tail}")
    return proc


def probe(video_path):
    """Return (duration_seconds, has_audio) for a media file."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    err = proc.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        raise ValueError(f"Not a readable media file: {video_path}")
    h, mnt, s = m.groups()
    duration = int(h) * 3600 + int(mnt) * 60 + float(s)
    has_audio = re.search(r"Stream #.*: Audio:", err) is not None
    return duration, has_audio


def extract_audio(video_path, wav_path, sample_rate, channels=2):
    _run(["-i", video_path, "-vn", "-ac", channels, "-ar", sample_rate,
          "-c:a", "pcm_s16le", wav_path])
    return Path(wav_path)


def _atempo_chain(factor):
    # atempo accepts 0.5–100 per filter in recent ffmpeg; chain to stay in 0.5–2.
    parts = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={p:.5f}" for p in parts)


def load_clip(path, sample_rate, tempo=1.0):
    """Decode any audio file to mono float32 at `sample_rate`, optionally
    changing its speed (pitch preserved)."""
    out = Path(path).with_suffix(".fit.wav")
    args = ["-i", path]
    if abs(tempo - 1.0) > 1e-3:
        args += ["-filter:a", _atempo_chain(tempo)]
    args += ["-ac", 1, "-ar", sample_rate, "-c:a", "pcm_f32le", out]
    _run(args)
    audio, _ = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)
    return audio


def audio_duration_any(path):
    """Duration in seconds of any audio file ffmpeg can decode (wav, mp3, ...)."""
    rate = 16000
    return len(load_clip(path, rate)) / rate


def mux(video_path, audio_wav, out_path, subtitles_path=None, subtitle_lang=None):
    """Replace the video's audio with `audio_wav`; the video stream is copied
    untouched. Optionally embed subtitles as a soft track."""
    args = ["-i", video_path, "-i", audio_wav]
    if subtitles_path:
        args += ["-i", subtitles_path]
    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if subtitles_path:
        args += ["-map", "2:s:0", "-c:s", "mov_text"]
        if subtitle_lang:
            args += ["-metadata:s:s:0", f"language={subtitle_lang}"]
    # No -shortest: the subtitle track ends at the last line and would cut the
    # video short. The mixed audio is already built to the video's length.
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out_path]
    _run(args)
    return Path(out_path)


def narrate(video_path, voice_path, out_path, original_volume=0.25):
    """Lay a narration track over a video: the original sound is kept but
    lowered to `original_volume`; the picture is copied untouched. The output
    keeps the video's length (narration longer than the video is cut)."""
    duration, has_audio = probe(video_path)
    if has_audio:
        graph = (f"[0:a]volume={original_volume}[bg];"
                 "[1:a]aresample=44100,apad[nar];"
                 "[bg][nar]amix=inputs=2:duration=first:normalize=0[mix]")
        args = ["-i", video_path, "-i", voice_path, "-filter_complex", graph,
                "-map", "0:v:0", "-map", "[mix]"]
    else:
        args = ["-i", video_path, "-i", voice_path, "-filter_complex",
                "[1:a]aresample=44100,apad[nar]", "-map", "0:v:0", "-map", "[nar]"]
    args += ["-t", f"{duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out_path]
    _run(args)
    return Path(out_path), duration


def write_wav(path, audio, sample_rate):
    sf.write(str(path), np.asarray(audio, dtype="float32"), sample_rate, subtype="PCM_16")
    return Path(path)

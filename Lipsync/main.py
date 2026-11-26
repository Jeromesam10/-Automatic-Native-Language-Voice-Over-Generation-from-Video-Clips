"""Lip sync a video to a voice track on its own (without the dubbing pipeline).

Typical use: re-run only the lip-sync step of a finished dubbing job:

  python Lipsync/main.py video.mp4 Dubbing/outputs/job_xxx/dubbed_voice.wav out.mp4

Writes a video-only file; mux it with the job's dubbed_audio.wav, or pass
--audio to have it muxed for you.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lipsync  # noqa: E402


def _ffmpeg():
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("voice_wav", help="voice-only track that drives the mouth")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0, help="first second to lip sync")
    ap.add_argument("--end", type=float, default=None, help="last second (default: end of video)")
    ap.add_argument("--audio", help="audio to put in the output (e.g. dubbed_audio.wav)")
    args = ap.parse_args()

    ffmpeg = _ffmpeg()
    end = args.end if args.end is not None else 10**9
    picture = Path(args.out).with_suffix(".video.mp4") if args.audio else Path(args.out)
    report = lipsync.lip_sync(args.video, args.voice_wav, picture, [(args.start, end)], ffmpeg=ffmpeg)
    print(json.dumps(report, indent=2))
    if report["applied"] and args.audio:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(picture),
                        "-i", args.audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", args.out], check=True)
        picture.unlink(missing_ok=True)
        print("Wrote", args.out)


if __name__ == "__main__":
    main()

import argparse
import json

from pipeline import dub_video


def main():
    parser = argparse.ArgumentParser(description="Dub a video into another language.")
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("target_language", help="Target language code, e.g. hi, ta, fr")
    parser.add_argument("--source", help="Source language code (auto-detected if omitted)")
    parser.add_argument("--tts", choices=["clone", "edge"],
                        help="clone = speaker's voice via Voice_Clone service; edge = stock voice")
    parser.add_argument("--reference", help="Speaker reference WAV (default: taken from the video)")
    parser.add_argument("--no-separate", action="store_true",
                        help="Skip Demucs; the background audio is not kept")
    parser.add_argument("--no-lipsync", action="store_true",
                        help="Skip lip sync (much faster on CPU)")
    args = parser.parse_args()

    result = dub_video(
        args.video,
        args.target_language,
        source_language=args.source,
        tts_backend=args.tts,
        speaker_reference=args.reference,
        separate=False if args.no_separate else None,
        lip_sync=False if args.no_lipsync else None,
    )
    summary = {k: v for k, v in result.items() if k != "segments"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

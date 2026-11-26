"""Command-line runner for narration mode (services must be running).

Usage: python main.py [video_path] [language] [speaker_reference]
"""
import sys

from graph import graph
from paths import resolve_speaker_reference, resolve_video

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/Video-525.mp4"
language = sys.argv[2] if len(sys.argv) > 2 else "ta"
reference = sys.argv[3] if len(sys.argv) > 3 else "voices/reference.wav"

initial_state = {
    "input": resolve_video(video_path),
    "language": language,
    "speaker_reference": resolve_speaker_reference(reference),
}

result = graph.invoke(initial_state)

print("\n==============================")
print("FINAL OUTPUT")
print("==============================")

print(result["final_output"])

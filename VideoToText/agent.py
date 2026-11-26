from video_reader import read_sampled_frames
from object_detector import detect_objects
from qwen_vision import describe_scene
from event_builder import build_events
from llm_reasoner import summarize


def run(video_path, frame_interval):
    print("========== VIDEO AGENT ==========")

    print("Reading video...")

    # Sample while decoding so only the frames we describe are kept in memory.
    selected_frames, total = read_sampled_frames(video_path, frame_interval)

    print(f"Total Frames : {total}")
    print(f"Selected Frames : {len(selected_frames)}")

    descriptions = []

    for frame in selected_frames:

        objects = detect_objects(frame)

        text = describe_scene(
            frame,
            objects
        )

        descriptions.append(text)

    events = build_events(descriptions)

    summary = summarize(events)

    return summary

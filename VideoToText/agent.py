from video_reader import read_video
from frame_selector import select_frames
from object_detector import detect_objects
from qwen_vision import describe_scene
from event_builder import build_events
from llm_reasoner import summarize


def run(video_path, frame_interval):
    print("========== VIDEO AGENT ==========")

    print("Reading video...")

    frames = read_video(video_path)

    print(f"Total Frames : {len(frames)}")

    selected_frames = select_frames(
        frames,
        frame_interval
    )

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
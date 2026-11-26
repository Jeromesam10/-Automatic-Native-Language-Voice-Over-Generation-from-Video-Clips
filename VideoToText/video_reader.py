import os

import cv2


def _open(video_path):
    print("=" * 50)
    print("VIDEO PATH")
    print(video_path)
    print("=" * 50)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video: {video_path}")

    return cap


def read_sampled_frames(video_path, interval):
    """Return (every `interval`-th frame, total frame count).

    Frames are sampled while decoding, so memory holds only the kept frames
    (a 1-minute 720p video is ~1,800 frames, ~5 GB if all were kept).
    """
    cap = _open(video_path)
    kept, total = [], 0
    try:
        while True:
            ok = cap.grab()                      # decode without copying out
            if not ok:
                break
            if total % interval == 0:
                ok, frame = cap.retrieve()
                if ok:
                    kept.append(frame)
            total += 1
    finally:
        cap.release()

    print("Frames :", total, "| kept:", len(kept))

    if not kept:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return kept, total


def read_video(video_path):
    """All frames (kept for backwards compatibility; prefer read_sampled_frames)."""
    frames, _ = read_sampled_frames(video_path, 1)
    return frames

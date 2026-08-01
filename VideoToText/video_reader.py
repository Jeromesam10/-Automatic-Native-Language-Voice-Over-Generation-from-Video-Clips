import cv2
import os

def read_video(video_path):

    print("=" * 50)
    print("VIDEO PATH")
    print(video_path)
    print("=" * 50)

    print("Exists :", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)

    print("Opened :", cap.isOpened())

    if not cap.isOpened():
        print("Unable to open video")

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    print("Frames :", len(frames))

    return frames
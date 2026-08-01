from PIL import Image
import imagehash
import cv2


def remove_duplicate_frames(frames, threshold=5):

    unique_frames = []

    previous_hash = None

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb)

        current_hash = imagehash.phash(img)

        if previous_hash is None:

            unique_frames.append(frame)

            previous_hash = current_hash

            continue

        difference = current_hash - previous_hash

        if difference > threshold:

            unique_frames.append(frame)

            previous_hash = current_hash

    return unique_frames
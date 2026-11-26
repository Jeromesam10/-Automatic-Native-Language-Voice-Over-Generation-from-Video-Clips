import os

from ultralytics import YOLO

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "yolo11x.pt"
)

model = YOLO(MODEL_PATH)


def detect_objects(frame):

    results = model(frame)

    objects = []

    for r in results:

        for c in r.boxes.cls:

            name = model.names[int(c)]

            objects.append(name)


    allowed = [

        "person",

        "car",

        "truck",

        "bus",

        "motorcycle",

        "bicycle",

        "dog",

        "cat",

        "chair",

        "table",

        "backpack"

    ]


    filtered = []


    for obj in objects:

        if obj in allowed:

            filtered.append(obj)


    return list(set(filtered))
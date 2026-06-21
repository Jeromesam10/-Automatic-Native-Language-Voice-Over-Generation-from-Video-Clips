from ultralytics import YOLO

model = YOLO("yolo11x.pt")


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
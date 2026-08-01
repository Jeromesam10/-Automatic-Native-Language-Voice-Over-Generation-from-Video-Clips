from transformers import pipeline
from PIL import Image
import cv2

pipe = None


def get_pipeline():
    global pipe

    if pipe is None:

        print("Loading Qwen Vision Model...")

        pipe = pipeline(
            "image-text-to-text",
            model="Qwen/Qwen2.5-VL-3B-Instruct"
        )

        print("Qwen Model Loaded Successfully.")

    return pipe


def describe_scene(frame, objects):

    model = get_pipeline()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    prompt = f"""
Detected Objects:
{', '.join(objects)}

Describe:
- What is happening.
- Important objects.
- Environment.
- Generate a short narration.
"""

    result = model(
        image,
        text=prompt,
        max_new_tokens=80
    )

    return result[0]["generated_text"]
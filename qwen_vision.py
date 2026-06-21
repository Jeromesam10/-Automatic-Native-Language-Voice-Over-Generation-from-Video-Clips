from transformers import pipeline
from PIL import Image
import cv2


pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen2.5-VL-3B-Instruct"
)


def describe_scene(frame, objects):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    prompt = f"""
    Detected Objects:

    {', '.join(objects)}

    Describe:

    - What is happening in the image.
    - Important objects.
    - Environment.
    - Generate a short human narration.
    """

    result = pipe(
        image,
        text=prompt,
        max_new_tokens=80
    )

    return result[0]["generated_text"]
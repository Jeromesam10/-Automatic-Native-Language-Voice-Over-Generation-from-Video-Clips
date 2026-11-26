from transformers import pipeline
from PIL import Image
import cv2

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

pipe = None


def get_pipeline():
    global pipe

    if pipe is None:

        print("Loading Qwen Vision Model...")

        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_ID
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

    detected = ", ".join(objects) if objects else "none"

    prompt = (
        f"Detected objects: {detected}.\n"
        "Write one or two plain sentences of voice-over narration describing "
        "what is happening in this scene and where it takes place. "
        "Return only the narration, with no headings, lists or labels."
    )

    # Chat-format input so the model sees the image token, and
    # return_full_text=False so the prompt is not echoed into the narration.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    result = model(
        text=messages,
        max_new_tokens=80,
        return_full_text=False
    )

    generated = result[0]["generated_text"]

    if isinstance(generated, list):
        # Some versions return the chat transcript; take the last message.
        generated = generated[-1].get("content", "")

    return str(generated).strip()

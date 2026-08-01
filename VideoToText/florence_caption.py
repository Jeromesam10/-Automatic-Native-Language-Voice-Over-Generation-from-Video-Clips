from transformers import pipeline
from PIL import Image
import cv2


pipe = pipeline(

    task="image-to-text",

    model="microsoft/Florence-2-base",

    trust_remote_code=True

)


def describe_frame(frame):

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )

    image = Image.fromarray(rgb)


    result = pipe(image)


    return result[0]["generated_text"]
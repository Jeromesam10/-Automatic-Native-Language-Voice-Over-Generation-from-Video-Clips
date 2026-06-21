from transformers import pipeline

from PIL import Image

import cv2


pipe = pipeline(

    "image-text-to-text",

    model="meta-llama/Llama-3.2-11B-Vision-Instruct"

)



def describe_scene(frame,objects):


    rgb=cv2.cvtColor(

    frame,

    cv2.COLOR_BGR2RGB

    )


    image=Image.fromarray(rgb)


    messages=[

    {

    "role":"user",

    "content":[

    {

    "type":"image",

    "image":image

    },

    {

    "type":"text",

    "text":

    f"""

    Detected Objects:

    {objects}


    Describe:

    1.What is happening?

    2.What are the important objects?

    3.Environment.

    4.Generate human narration.


    """

    }

    ]

    }

    ]


    output=pipe(

    messages,

    max_new_tokens=120

    )


    return output[0]["generated_text"]
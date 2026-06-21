from config import VIDEO_PATH, FRAME_INTERVAL, OUTPUT_FILE

from video_reader import read_video

from frame_selector import select_frames

from object_detector import detect_objects

from qwen_vision import describe_scene

from event_builder import build_events

from llm_reasoner import summarize



print("Reading video...")


frames=read_video(

VIDEO_PATH

)


print(

"Total frames:",

len(frames)

)



selected_frames=select_frames(

frames,

FRAME_INTERVAL

)



print(

"Selected frames:",

len(selected_frames)

)



descriptions=[]



for frame in selected_frames:


    objects=detect_objects(frame)


    print("\nObjects:")

    print(objects)



    text=describe_scene(

        frame,

        objects

    )



    print("\nDescription:")

    print(text)



    descriptions.append(text)




events=build_events(

descriptions

)



summary=summarize(

events

)



print("\nFINAL DESCRIPTION:\n")


print(summary)



with open(

OUTPUT_FILE,

"w",

encoding="utf-8"

) as f:


    f.write(summary)
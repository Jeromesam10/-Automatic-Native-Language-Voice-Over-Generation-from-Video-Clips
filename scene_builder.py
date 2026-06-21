def build_scene(objects):

    unique_objects = list(set(objects))

    sentence = "The video contains "

    sentence += ", ".join(unique_objects)

    return sentence

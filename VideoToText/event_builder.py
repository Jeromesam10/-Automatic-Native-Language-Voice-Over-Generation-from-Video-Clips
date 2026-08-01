from collections import Counter

def build_events(descriptions):

    unique=[]

    for d in descriptions:

        if d not in unique:

            unique.append(d)

    return unique
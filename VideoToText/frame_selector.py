def select_frames(frames, interval):

    selected=[]

    for i in range(

        0,

        len(frames),

        interval

    ):

        selected.append(frames[i])

    return selected
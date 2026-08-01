from convert import subtitles_to_speech
from pathlib import Path
from convert import subtitles_to_speech


def run(
    subtitle_file,
    speaker_reference,
    language=None
):

    output = subtitles_to_speech(
        subtitles_path=subtitle_file,
        output_path="outputs/final_voice.wav",
        speaker_wav=speaker_reference,
        language=language
    )

    return {
        "cloned_audio": str(output)
    }
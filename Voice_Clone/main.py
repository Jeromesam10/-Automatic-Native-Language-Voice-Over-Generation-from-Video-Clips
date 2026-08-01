from agent import run

result = run(
    subtitle_file="subtitles.txt",
    speaker_reference="voices/reference.wav",
    language="ta"
)

print(result)
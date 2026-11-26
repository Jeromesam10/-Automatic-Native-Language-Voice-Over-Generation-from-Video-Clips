from agent import run

text = input("Enter text: ")

language = input("""
Choose language

ta - Tamil
hi - Hindi
te - Telugu
ml - Malayalam
fr - French
es - Spanish

Enter code:
""")

result = run(
    text,
    language,
    output_file="translated_audio.mp3"
)

print("\nTranslated Text:")
print(result["translated_text"])

print("\nAudio saved successfully!")

print("Location:")
print(result["audio_path"])
import os

from config import VIDEO_PATH, FRAME_INTERVAL, OUTPUT_FILE
from agent import run

print("Program Started")

summary = run(VIDEO_PATH, FRAME_INTERVAL)

print("\n========== FINAL SUMMARY ==========\n")
print(summary)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Saved to {OUTPUT_FILE}")
print("Program Finished")

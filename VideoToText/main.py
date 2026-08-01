from config import VIDEO_PATH, FRAME_INTERVAL
from agent import run

summary = run(VIDEO_PATH, FRAME_INTERVAL)

print("\n========== FINAL SUMMARY ==========\n")
print(summary)

print("Program Started")

from config import VIDEO_PATH, FRAME_INTERVAL
print("Config Imported")

from agent import run
print("Agent Imported")

summary = run(VIDEO_PATH, FRAME_INTERVAL)

print(summary)

print("Program Finished")
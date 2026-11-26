"""Run the VideoToText pipeline from the project root.

The pipeline modules live in VideoToText/ and use plain imports
(`from config import ...`), so put that folder on the import path first.

For the full video -> translation -> cloned voice chain, start the three
services and use Langraph_Orchestrater/main.py instead.
"""
import os
import runpy
import sys

VIDEO_TO_TEXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VideoToText")

sys.path.insert(0, VIDEO_TO_TEXT_DIR)

runpy.run_path(os.path.join(VIDEO_TO_TEXT_DIR, "main.py"), run_name="__main__")

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Services run on the same machine and exchange absolute file paths.
VIDEO_TO_TEXT_DIR = PROJECT_ROOT / "VideoToText"
VOICE_CLONE_DIR = PROJECT_ROOT / "Voice_Clone"

# Service endpoints (override with environment variables when the services
# run on other hosts/ports).
VIDEO_API_URL = os.getenv("VIDEO_API_URL", "http://127.0.0.1:8002/video")
TRANSLATOR_API_URL = os.getenv("TRANSLATOR_API_URL", "http://127.0.0.1:8000/translate")
VOICE_API_URL = os.getenv("VOICE_API_URL", "http://127.0.0.1:8001/voice")
DUBBING_API_URL = os.getenv("DUBBING_API_URL", "http://127.0.0.1:8003/dub")
NARRATE_API_URL = os.getenv("NARRATE_API_URL", "http://127.0.0.1:8003/narrate")

DEFAULT_LANGUAGE = os.getenv("TARGET_LANGUAGE", "ta")
DEFAULT_SPEAKER_REFERENCE = os.getenv("SPEAKER_REFERENCE", "voices/reference.wav")

# Model inference is slow (especially on CPU), so allow generous timeouts.
# Values are (connect, read) seconds.
VIDEO_TIMEOUT = (10, float(os.getenv("VIDEO_TIMEOUT", "1800")))
TRANSLATOR_TIMEOUT = (10, float(os.getenv("TRANSLATOR_TIMEOUT", "120")))
VOICE_TIMEOUT = (10, float(os.getenv("VOICE_TIMEOUT", "1800")))
DUBBING_TIMEOUT = (10, float(os.getenv("DUBBING_TIMEOUT", "7200")))
NARRATE_TIMEOUT = (10, float(os.getenv("NARRATE_TIMEOUT", "600")))

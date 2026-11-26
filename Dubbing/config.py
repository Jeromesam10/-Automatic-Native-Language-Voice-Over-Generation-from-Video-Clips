import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("DUBBING_OUTPUT_DIR", BASE_DIR / "outputs"))

# Working sample rate for the mixed soundtrack (Demucs works at 44.1 kHz).
SAMPLE_RATE = 44100

# --- Speech recognition (faster-whisper) ---
# tiny / base / small / medium / large-v3. Bigger = more accurate, slower.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")          # auto / cpu / cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "default")

# --- Vocal / background separation (Demucs) ---
SEPARATE_VOCALS = os.getenv("SEPARATE_VOCALS", "1") == "1"
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")

# --- Voice synthesis ---
# "clone": call the Voice_Clone service (voice cloned from the video's speaker)
# "edge":  stock Microsoft Edge neural voice, no cloning (fast, for testing)
TTS_BACKEND = os.getenv("TTS_BACKEND", "clone")
VOICE_API_URL = os.getenv("VOICE_API_URL", "http://127.0.0.1:8001/voice")
VOICE_TIMEOUT = (10, float(os.getenv("VOICE_TIMEOUT", "900")))

# Seconds of the speaker's own (separated) voice used as the cloning reference.
# Below MIN cloning is refused; below RECOMMENDED it works but sounds less
# like the speaker (a warning is added to the result).
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_RECOMMENDED_SECONDS = 6.0
REFERENCE_MAX_SECONDS = 20.0

# --- Timing fit ---
# A dubbed line longer than its slot is sped up, at most by MAX_SPEEDUP.
# Anything still too long is faded out at the start of the next line.
MAX_SPEEDUP = float(os.getenv("MAX_SPEEDUP", "1.35"))

# Segments closer than this (seconds) are merged so TTS gets whole sentences.
MERGE_GAP_SECONDS = 0.35
MERGE_MAX_SECONDS = 12.0

# --- Lip sync (MuseTalk 1.5, see Lipsync/README.md) ---
# Re-renders the speaker's mouth to match the dubbed speech. Skipped
# automatically when there is no clear on-screen face. Slow on CPU.
LIP_SYNC = os.getenv("LIP_SYNC", "1") == "1"
LIPSYNC_DIR = BASE_DIR.parent / "Lipsync"

# Loudness of the dubbed voice relative to the background.
VOICE_GAIN = 1.0
BACKGROUND_GAIN = 0.9

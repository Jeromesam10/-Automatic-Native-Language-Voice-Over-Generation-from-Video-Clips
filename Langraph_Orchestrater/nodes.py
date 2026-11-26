import requests

from config import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER_REFERENCE,
    NARRATE_API_URL,
    NARRATE_TIMEOUT,
    TRANSLATOR_API_URL,
    TRANSLATOR_TIMEOUT,
    VIDEO_API_URL,
    VIDEO_TIMEOUT,
    VOICE_API_URL,
    VOICE_TIMEOUT,
)
from state import WorkflowState


class ServiceError(Exception):
    """Raised when a downstream agent service fails.

    `status` is the HTTP status to return to our caller: a downstream 4xx
    (bad input, e.g. "no speech in this video") stays a 4xx; anything else
    (service down, crash) becomes 502.
    """

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def _post(name, url, payload, timeout):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ServiceError(f"{name} unreachable at {url}: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        status = response.status_code if 400 <= response.status_code < 500 else 502
        raise ServiceError(f"{name} error {response.status_code}: {detail}", status)

    return response.json()


def _language(state: WorkflowState):
    return state.get("language") or DEFAULT_LANGUAGE


# -------------------------
# Agent 1 - Video Agent
# -------------------------

def agent1(state: WorkflowState):

    print("\n========== VIDEO AGENT ==========")
    print("Video:", state["input"])

    result = _post(
        "Video API",
        VIDEO_API_URL,
        {"video_path": state["input"]},
        VIDEO_TIMEOUT,
    )

    caption = (result.get("caption") or "").strip()
    if not caption:
        raise ServiceError("Video API returned an empty caption")

    print("Caption :", caption)

    return {"agent1_output": caption}


# -------------------------
# Agent 2 - Translator
# -------------------------

def agent2(state: WorkflowState):

    print("\n========== TRANSLATOR AGENT ==========")

    result = _post(
        "Translator API",
        TRANSLATOR_API_URL,
        {
            "text": state["agent1_output"],
            "language": _language(state),
            "source_language": "en",   # VideoToText captions are English
        },
        TRANSLATOR_TIMEOUT,
    )

    print("Translated Text :", result["translated_text"])
    print("Preview Audio   :", result.get("audio_path") or result.get("audio_error"))

    return {
        "translated_text": result["translated_text"],
        "translated_audio": result.get("audio_path"),
    }


# -------------------------
# Agent 3 - Voice Clone
# -------------------------

def agent3(state: WorkflowState):

    print("\n========== VOICE CLONE AGENT ==========")

    result = _post(
        "Voice Clone API",
        VOICE_API_URL,
        {
            "translated_text": state["translated_text"],
            "speaker_reference": state.get("speaker_reference")
            or DEFAULT_SPEAKER_REFERENCE,
            "language": _language(state),
        },
        VOICE_TIMEOUT,
    )

    voice_output = result["voice_path"]

    print("Voice File :", voice_output)

    return {"voice_output": voice_output}


# -------------------------
# Agent 4 - Compose narrated video
# -------------------------

def agent4(state: WorkflowState):

    print("\n========== VIDEO COMPOSE AGENT ==========")

    result = _post(
        "Narrate API",
        NARRATE_API_URL,
        {
            "video_path": state["input"],
            "voice_path": state["voice_output"],
        },
        NARRATE_TIMEOUT,
    )

    print("Narrated Video :", result["narrated_video"])

    return {
        "narrated_video": result["narrated_video"],
        "final_output": {
            "narrated_video": result["narrated_video"],
            "caption": state["agent1_output"],
            "translated_text": state["translated_text"],
            "translated_audio": state.get("translated_audio"),
            "voice_output": state["voice_output"],
            "warnings": result.get("warnings", []),
        },
    }

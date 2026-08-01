from state import WorkflowState
import requests


# -------------------------
# Agent 1 - Video Agent
# -------------------------

def agent1(state: WorkflowState):

    print("\n========== VIDEO AGENT ==========")
    print("Input Received:", state["input"])

    # ---------- OPTION 1 ----------
    # Dummy output (current)
    state["agent1_output"] = state["input"]

    # ---------- OPTION 2 ----------
    # When Video API is ready, replace the above line with:
    #
    # response = requests.post(
    #     "http://127.0.0.1:8002/video",
    #     json={
    #         "video_path": state["input"]
    #     }
    # )
    #
    # result = response.json()
    #
    # state["agent1_output"] = result["caption"]

    print("Caption :", state["agent1_output"])

    return state


# -------------------------
# Agent 2 - Translator
# -------------------------

def agent2(state: WorkflowState):

    print("\n========== TRANSLATOR AGENT ==========")

    response = requests.post(
        "http://127.0.0.1:8000/translate",
        json={
            "text": state["agent1_output"],
            "language": "ta"
        }
    )

    if response.status_code != 200:
        raise Exception(
            f"Translator API Error\n{response.text}"
        )

    result = response.json()

    state["translated_text"] = result["translated_text"]
    state["translated_audio"] = result["audio_path"]

    print("Translated Text :", state["translated_text"])
    print("Audio File      :", state["translated_audio"])

    return state


# -------------------------
# Agent 3 - Voice Clone
# -------------------------

def agent3(state: WorkflowState):

    print("\n========== VOICE CLONE AGENT ==========")

    response = requests.post(
        "http://127.0.0.1:8001/voice",
        json={
            "translated_text": state["translated_text"],
            "speaker_reference": "voices/reference.wav",
            "language": "ta"
        }
    )

    if response.status_code != 200:
        raise Exception(
            f"Voice Clone API Error\n{response.text}"
        )

    result = response.json()

    state["voice_output"] = result["voice_path"]

    state["final_output"] = {
        "caption": state["agent1_output"],
        "translated_text": state["translated_text"],
        "translated_audio": state["translated_audio"],
        "voice_output": state["voice_output"]
    }

    print("Voice File :", state["voice_output"])

    return state
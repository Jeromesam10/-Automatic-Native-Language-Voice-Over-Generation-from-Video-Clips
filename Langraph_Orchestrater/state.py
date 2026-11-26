from typing import TypedDict


class WorkflowState(TypedDict, total=False):

    # Inputs
    input: str              # path to the video (absolute, or relative to VideoToText/)
    language: str           # target language code, e.g. "ta"
    speaker_reference: str  # path to reference voice (relative to Voice_Clone/)

    # Agent outputs
    agent1_output: str

    translated_text: str

    translated_audio: str

    voice_output: str

    narrated_video: str

    final_output: dict

from typing import TypedDict


class WorkflowState(TypedDict):

    input: str

    agent1_output: str

    translated_text: str

    translated_audio: str

    voice_output: str

    final_output: dict
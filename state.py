from typing import TypedDict

class WorkflowState(TypedDict):
    input: str
    agent1_output: str
    agent2_output: str
    final_output: str
from langgraph.graph import StateGraph, END

from state import WorkflowState
from nodes import agent1, agent2, agent3, agent4


# Narration mode:
#   Agent1 describe video -> Agent2 translate -> Agent3 clone voice
#   -> Agent4 lay the voice-over onto the video
workflow = StateGraph(WorkflowState)

workflow.add_node("Agent1", agent1)
workflow.add_node("Agent2", agent2)
workflow.add_node("Agent3", agent3)
workflow.add_node("Agent4", agent4)

workflow.set_entry_point("Agent1")

workflow.add_edge("Agent1", "Agent2")
workflow.add_edge("Agent2", "Agent3")
workflow.add_edge("Agent3", "Agent4")
workflow.add_edge("Agent4", END)

graph = workflow.compile()

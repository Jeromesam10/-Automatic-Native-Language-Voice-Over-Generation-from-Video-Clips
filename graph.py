from langgraph.graph import StateGraph, END

from state import WorkflowState
from nodes import agent1, agent2, agent3


workflow = StateGraph(WorkflowState)

workflow.add_node("Agent1", agent1)
workflow.add_node("Agent2", agent2)
workflow.add_node("Agent3", agent3)

workflow.set_entry_point("Agent1")

workflow.add_edge("Agent1", "Agent2")
workflow.add_edge("Agent2", "Agent3")
workflow.add_edge("Agent3", END)

graph = workflow.compile()
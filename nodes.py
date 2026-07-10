from state import WorkflowState


def agent1(state: WorkflowState):
    print("\n========== AGENT 1 ==========")
    print("Input Received:", state["input"])

    state["agent1_output"] = f"Processed by Agent 1 -> {state['input']}"

    print("Output:", state["agent1_output"])

    return state


def agent2(state: WorkflowState):
    print("\n========== AGENT 2 ==========")
    print("Input Received:", state["agent1_output"])

    state["agent2_output"] = f"Processed by Agent 2 -> {state['agent1_output']}"

    print("Output:", state["agent2_output"])

    return state


def agent3(state: WorkflowState):
    print("\n========== AGENT 3 ==========")
    print("Input Received:", state["agent2_output"])

    state["final_output"] = (
        f"Final Output Generated -> {state['agent2_output']}"
    )

    print("Output:", state["final_output"])

    return state
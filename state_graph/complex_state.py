from typing import TypedDict
from langgraph.graph import END, StateGraph, START


class SimpleState(TypedDict):

    count: int
    sum: int


def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    return {"count": new_count, "sum": state["sum"] + new_count}


def should_continue(state: SimpleState):

    if state["count"] < 5:
        return "continue"
    return "stop"


graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_conditional_edges(
    "increment", should_continue, {"continue": "increment", "stop": END}
)

app = graph.compile()

state = {"count": 0, "sum": 0}

result = app.invoke(state)
print(result)

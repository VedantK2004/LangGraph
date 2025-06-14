from typing import List, Union

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revised_instruction_chain, first_responder_chain
from execute_tool import execute_tools

graph = MessageGraph()
MAX_ITERATIONS = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", execute_tools)
graph.add_node("revisor", revised_instruction_chain)


graph.set_entry_point("draft")
graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


graph.add_conditional_edges(
    "revisor",
    event_loop,
    {"execute_tools": "execute_tools", "draft": "draft", END: END},
)


app = graph.compile()

print(app.get_graph().draw_mermaid())
print(app.get_graph().draw_ascii())

response = app.invoke(
    {
        "role": "user",
        "content": "What is the capital of France? Please provide a detailed answer with references.",
    }
)

print(response["tool_calls"][0]["args"]["answer"])
print(response, "response")

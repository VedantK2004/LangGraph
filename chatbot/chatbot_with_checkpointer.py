from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
memory = MemorySaver()


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):

    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config: RunnableConfig = {
    "configurable": {"thread_id": "1"}
}  # Fixed key and value type

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    else:
        response = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print("Bot: ", response["messages"][-1].content)

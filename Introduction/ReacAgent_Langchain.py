from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# print(llm.invoke("Give me the tweet about today's weather in Pune.").content)

search_tool = [TavilySearchResults(search_depth="basic")]

agent = initialize_agent(
    tools=search_tool,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

response = agent.invoke({"input": "What is the full for of NASA?"})
print(response["output"])

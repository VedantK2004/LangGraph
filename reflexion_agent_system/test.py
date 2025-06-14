from pydantic import SecretStr
from langchain_openai import ChatOpenAI
import os

api = os.getenv("OPENROUTER_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatOpenAI(
    api_key=SecretStr(api) if api is not None else None,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-small-24b-instruct-2501:free",
)

print(llm.invoke("what should I do today?").content)

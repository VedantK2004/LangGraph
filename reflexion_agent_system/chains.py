from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import datetime
from dotenv import load_dotenv
from schema import AnswerQuestion, RevisedAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
import os

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

load_dotenv()
api = os.getenv("OPENROUTER_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatOpenAI(
    api_key=SecretStr(api) if api is not None else None,
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3.3-70b-instruct:free",
)

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI Researcher. Current time: {current_time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize your improvement.
            3. After thr reflection, **List 2-3 search queries separately** for researching improvements. Do not include them inside the reflection.""",
        ),
        MessagesPlaceholder(variable_name="messages"),  # For user message or query
        ("system", "Answer the user's above questions using the required format."),
    ]
).partial(current_time=lambda: datetime.datetime.now().isoformat())

revise_instructions = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You must include numerical citations in your answer to ensure it can be verified.
- Add a "References" section to the bottom of you answer (which does not count towards the word limit). In form of:
    - [1] https//:example.com
    - [2] https//:example.com
- You should use the previous critique to remove the superfluous information from your answer and make sure it is not more that 250 words.
"""

first_responder_promt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words answer."
)

first_responder_chain = first_responder_promt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

validator = PydanticToolsParser(tools=[AnswerQuestion])

revised_instruction_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[RevisedAnswer], tool_choice="RevisedAnswer")


response = first_responder_chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Write me a blog post related to how AI growth impacts IT industry."
            )
        ]
    }
)

print(response)

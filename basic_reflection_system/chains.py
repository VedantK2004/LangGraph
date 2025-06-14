from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer tasked with writing excellent twitter posts."
            "Generat the best twitter post possible for the user's request."
            "If the user provides critique, respond with your revised version of your previous attempt",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requets for the length, virality,and style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

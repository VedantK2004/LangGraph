from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(description="~250 words detailed anwser to the question.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    reflection: Reflection = Field(description="Your reflection on the initial answer.")


class RevisedAnswer(BaseModel):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answers."
    )

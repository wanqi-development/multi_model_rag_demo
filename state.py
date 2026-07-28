from typing import Literal

from langchain.agents import AgentState
from langgraph.graph import MessagesState


class GraphRagState(MessagesState):
    input_type: Literal["only_text", "only_image", "text_and_image"]
    input_image: str
    input_text: str
    retrieved_docs: list[str]
    retrieved_images: list[str]
    answer: str
    precision: float
    current_user: str
    approve_result: Literal["agree", "reject"]

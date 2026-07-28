from enum import StrEnum

from pydantic import BaseModel

class MessageType(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"



class ChatContext(BaseModel):

    def __init__(self, context: str, message_type: MessageType, user: str, timestamp: int, context_dense: list[float]):
        super().__init__()
        self.context = context
        self.message_type = message_type.value
        self.user = user
        self.timestamp = timestamp
        self.context_dense = context_dense


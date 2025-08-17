from pydantic import BaseModel, constr

from typing import Literal

class ChatRequest(BaseModel):
    message: constr(min_length=1, max_length=2000)
    provider: Literal["openai", "gemini"]

class ChatResponse(BaseModel):
    reply: str
    latency_ms: int

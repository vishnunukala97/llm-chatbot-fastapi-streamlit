from pydantic import BaseModel, constr

class ChatRequest(BaseModel):
    message: constr(min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    reply: str
    latency_ms: int

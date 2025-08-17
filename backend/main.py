"""
FastAPI application exposing:
- GET /health         : liveness
- POST /chat          : accepts {"message": "..."}; optional X-Session-Id header
- GET /history        : returns current session history (for debugging/UI)
- POST /clear-session : clears server-side history and rate limit state

Security:
- Secrets remain on the server (LLMService holds API keys).
- Input validated via Pydantic and by LLMService.
"""

from typing import List, Optional
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from backend.services.llm import LLMService

app = FastAPI(title="Basic LLM Chatbot (Gemini)")
llm = LLMService()  # single instance reused across requests


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    total_history_turns: Optional[int] = None


class HistoryMessage(BaseModel):
    role: str
    content: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_session_id: str | None = Header(default=None)):
    """
    - Accepts a user message.
    - Optional X-Session-Id enables per-session history & rate limiting.
    - Returns assistant reply + light telemetry for UI display.
    """
    # Use the richer API to get telemetry
    result = llm.chat(req.message, session_id=x_session_id)
    return ChatResponse(
        reply=result.reply,
        model=result.model,
        latency_ms=result.latency_ms,
        total_history_turns=result.total_history_turns,
    )


@app.get("/history", response_model=List[HistoryMessage])
def get_history(x_session_id: str | None = Header(default=None)):
    """
    Fetch server-side conversation history for the current session.
    Useful for debugging, tests, or a 'show history' UI.
    """
    if not x_session_id:
        return []
    history = llm.get_history(x_session_id)
    return [HistoryMessage(**m) for m in history]


@app.post("/clear-session")
def clear_session(x_session_id: str | None = Header(default=None)):
    """
    Clear server-side session history and rate-limit state.
    """
    if x_session_id:
        llm.clear_session(x_session_id)
    return {"cleared": bool(x_session_id)}

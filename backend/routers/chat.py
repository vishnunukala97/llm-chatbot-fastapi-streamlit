from __future__ import annotations

import logging, time
import httpx
from tenacity import RetryError
from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.chat import ChatIn, ChatOut
from ..services.llm import LLMService, TransientLLMError
from backend.core.exceptions import (
    RateLimitException,
    ProviderUnavailableException,
    UpstreamTimeoutException,
    ChatAPIException,
    raise_for_status,
)

# Import limiter from main
from backend.main import limiter

log = logging.getLogger("routers.chat")
router = APIRouter(tags=["chat"])

def get_llm_service(request: Request) -> LLMService:
    svc = getattr(request.app.state, "llm_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized.")
    return svc

@router.post("/chat", response_model=ChatOut, summary="Send a prompt and receive a model reply")
@limiter.limit("10/minute")
async def chat(payload: ChatIn, svc: LLMService = Depends(get_llm_service)) -> ChatOut:
    t0 = time.perf_counter()
    try:
        reply = await svc.generate(payload.message)
    except HTTPException:
        # Already mapped precisely in the service
        raise
    except RetryError as e:
        last = e.last_attempt.exception()
        if isinstance(last, TransientLLMError):
            code = last.status_code
            raise_for_status(code)
        if isinstance(last, httpx.TimeoutException):
            raise_for_status(504)
        if isinstance(last, httpx.HTTPError):
            raise_for_status(502, "Network error talking to provider.")
        raise_for_status(502, "Upstream error after retries.")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return ChatOut(reply=reply, latency_ms=latency_ms)

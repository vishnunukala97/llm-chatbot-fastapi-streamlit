from __future__ import annotations

import logging, time
import httpx
from tenacity import RetryError
from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.chat import ChatIn, ChatOut
from ..services.llm import LLMService, TransientLLMError

log = logging.getLogger("routers.chat")
router = APIRouter(tags=["chat"])

def get_llm_service(request: Request) -> LLMService:
    svc = getattr(request.app.state, "llm_service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized.")
    return svc

@router.post("/chat", response_model=ChatOut, summary="Send a prompt and receive a model reply")
async def chat(payload: ChatIn, svc: LLMService = Depends(get_llm_service)) -> ChatOut:
    t0 = time.perf_counter()
    try:
        reply = await svc.generate(payload.message)
    except HTTPException:
        # Already mapped precisely in the service
        raise
    except RetryError as e:
        # Retries exhausted: map the root cause
        last = e.last_attempt.exception()
        if isinstance(last, TransientLLMError):
            code = last.status_code
            if code == 428:
                raise HTTPException(status_code=429, detail="Rate limit from provider. Please retry shortly.")
            if code in (503,):
                raise HTTPException(status_code=503, detail="Provider overloaded/unavailable.")
            if code in (504,):
                raise HTTPException(status_code=504, detail="Upstream timeout.")
            raise HTTPException(status_code=502, detail="Transient upstream error after retries.")
        if isinstance(last, httpx.TimeoutException):
            raise HTTPException(status_code=504, detail="Upstream timeout.")
        if isinstance(last, httpx.HTTPError):
            raise HTTPException(status_code=502, detail="Network error talking to provider.")
        # Fallback
        raise HTTPException(status_code=502, detail="Upstream error after retries.")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return ChatOut(reply=reply, latency_ms=latency_ms)

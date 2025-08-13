import time
import httpx
import logging
from fastapi import APIRouter, Depends, HTTPException
from ..schemas.chat import ChatIn, ChatOut
from ..clients.gemini import generate
from ..core.config import settings

log = logging.getLogger("routers.chat")
router = APIRouter(prefix="", tags=["chat"])

# Dependency to get a shared httpx.AsyncClient from app state
async def get_http_client() -> httpx.AsyncClient:
    # This will be set in main.lifespan
    from ..main import http_client
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized.")
    return http_client

@router.post("/chat", response_model=ChatOut)
async def chat(payload: ChatIn, client: httpx.AsyncClient = Depends(get_http_client)) -> ChatOut:
    t0 = time.perf_counter()
    reply = await generate(payload.message, client)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return ChatOut(reply=reply, latency_ms=elapsed_ms)

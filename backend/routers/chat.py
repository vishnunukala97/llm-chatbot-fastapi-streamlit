from fastapi import APIRouter, Depends, HTTPException, status, Request
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.llm import LLMClient, LLMError, LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMResponseError, LLMServerError
from backend.core.logging import setup_logging
import httpx
import time
import html


import logging
router = APIRouter()
log = logging.getLogger("chat")

async def get_llm_client() -> LLMClient:
    async with httpx.AsyncClient() as client:
        yield LLMClient(client)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    llm_client: LLMClient = Depends(get_llm_client)
):
    # Input sanitation (basic, see P2 for advanced)
    user_message = html.escape(req.message.strip())
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    start = time.perf_counter()
    try:
        reply = await llm_client.generate_reply(user_message, provider=req.provider)
        latency = int((time.perf_counter() - start) * 1000)
        return ChatResponse(reply=reply, latency_ms=latency)
    except LLMAuthError as e:
        log.warning({"event": "auth_error", "error": str(e)})
        raise HTTPException(status_code=401, detail="Authentication failed for LLM provider.")
    except LLMRateLimitError as e:
        log.warning({"event": "rate_limit", "error": str(e)})
        raise HTTPException(status_code=429, detail="LLM provider rate limit exceeded. Please try again later.")
    except LLMTimeoutError as e:
        log.error({"event": "timeout", "error": str(e)})
        raise HTTPException(status_code=504, detail="LLM provider timed out. Please try again.")
    except LLMResponseError as e:
        log.error({"event": "response_error", "error": str(e)})
        raise HTTPException(status_code=502, detail="Invalid response from LLM provider.")
    except LLMServerError as e:
        log.error({"event": "server_error", "error": str(e)})
        raise HTTPException(status_code=502, detail="LLM provider server error.")
    except LLMError as e:
        log.error({"event": "llm_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="LLM error: " + str(e))
    except Exception as e:
        log.error({"event": "unexpected_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Unexpected server error.")

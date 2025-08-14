from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import setup_logging
from .routers.chat import router as chat_router
from .services.llm import GeminiService

# --- Rate limiting imports ---
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse

# Initialize Limiter globally
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifecycle:
      - configure logging
      - create shared HTTP client
      - create LLM service
      - store both on app.state
      - cleanly close resources on shutdown
    """
    setup_logging()
    logging.getLogger("uvicorn.error").setLevel(settings.LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(settings.LOG_LEVEL)

    # Shared HTTP client (HTTP/1.1; no h2 extra needed)
    http_client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)
    app.state.http_client = http_client

    # Service uses env-loaded API key/model
    app.state.llm_service = GeminiService(client=http_client)

    try:
        yield
    finally:
        await http_client.aclose()
        app.state.http_client = None
        app.state.llm_service = None

app = FastAPI(title="LLM Chatbot API", version="1.0.0", lifespan=lifespan)

# Add SlowAPI middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Add rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit from provider—please retry shortly."},
    )

# CORS: allow Streamlit on localhost by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
)

# Routes
app.include_router(chat_router)

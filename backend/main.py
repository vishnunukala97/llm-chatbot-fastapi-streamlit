from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from .core.config import settings
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.logging import setup_logging
from .routers.chat import router as chat_router

# will be set in lifespan; imported by router dependency
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    setup_logging()
    logging.getLogger("uvicorn.error").setLevel(settings.LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(settings.LOG_LEVEL)

    # Shared HTTP client
    http_client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)
    try:
        yield
    finally:
        await http_client.aclose()
        http_client = None

app = FastAPI(title="LLM Chatbot API", version="1.0.0", lifespan=lifespan)

# CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routes
app.include_router(chat_router)

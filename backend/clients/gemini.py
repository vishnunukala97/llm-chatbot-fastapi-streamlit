from __future__ import annotations
import json, logging
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from fastapi import HTTPException
from ..core.config import settings

log = logging.getLogger("clients.gemini")

TRANSIENT = {429, 500, 502, 503, 504}

class TransientLLMError(Exception):
    """Retryable upstream errors."""

def _build_url() -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
    )

@retry(
    reraise=True,
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, max=12),
    retry=retry_if_exception_type((TransientLLMError, httpx.HTTPError)),
)
async def generate(prompt: str, client: httpx.AsyncClient) -> str:
    """Call Gemini and return text; raise HTTPException on non-retryable failures."""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")

    url = _build_url()
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        resp = await client.post(url, json=payload, timeout=settings.REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("HTTP error to Gemini: %s", e)
        raise

    # Retry transient HTTP status codes
    if resp.status_code in TRANSIENT:
        snippet = resp.text[:200]
        log.warning("Transient %s from Gemini: %s", resp.status_code, snippet)
        raise TransientLLMError(f"Transient upstream error {resp.status_code}")

    if resp.status_code == 401 or resp.status_code == 403:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized API key.")
    if not resp.is_success:
        # Non-retryable upstream failure
        raise HTTPException(status_code=502, detail=f"Provider error: {resp.text[:200]}")

    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    except Exception:
        raise HTTPException(status_code=502, detail="Malformed provider response.")
    if not text:
        raise HTTPException(status_code=502, detail="Empty provider response.")
    return text

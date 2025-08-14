from __future__ import annotations

from typing import Protocol, Optional
import logging
import httpx
from tenacity import (
    retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
)
from fastapi import HTTPException
from ..core.config import settings

log = logging.getLogger("services.llm")
_TRANSIENT = {429, 500, 502, 503, 504}


class TransientLLMError(Exception):
    """Retryable upstream errors (carry status code + short detail)."""
    def __init__(self, status_code: int, detail: str = "") -> None:
        super().__init__(f"Transient upstream error {status_code}")
        self.status_code = status_code
        self.detail = detail


class LLMService(Protocol):
    async def generate(self, prompt: str) -> str: ...


def _get_api_key() -> Optional[str]:
    # Works whether GEMINI_API_KEY is a str or SecretStr
    key = getattr(settings, "GEMINI_API_KEY", None)
    if hasattr(key, "get_secret_value"):
        return key.get_secret_value()  # type: ignore[attr-defined]
    return key


class GeminiService:
    """
    Service class that encapsulates LLM calls (Gemini).
    - API key/model loaded from environment via `settings`
    - Timeouts/retries baked in
    """
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        request_timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.client = client
        self.api_key = api_key or _get_api_key()
        self.model = model or settings.GEMINI_MODEL
        self.request_timeout = request_timeout or settings.REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.MAX_RETRIES

    def _url(self) -> str:
        if not self.api_key:
            # Let generate() raise a clean 401 instead
            return ""
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_random_exponential(multiplier=1, max=12),
        retry=retry_if_exception_type((TransientLLMError, httpx.HTTPError)),
    )
    async def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise HTTPException(status_code=401, detail="Missing API key.")
        if not prompt or not prompt.strip():
            raise HTTPException(status_code=400, detail="Message is empty.")

        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        try:
            resp = await self.client.post(self._url(), json=payload, timeout=self.request_timeout)
        except httpx.HTTPError as e:
            log.warning("HTTP error to Gemini: %s", e)
            raise

        if resp.status_code in _TRANSIENT:
            snippet = resp.text[:200]
            log.warning("Transient %s from Gemini: %s", resp.status_code, snippet)
            raise TransientLLMError(resp.status_code, snippet)

        if resp.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Invalid or unauthorized API key.")

        if not resp.is_success:
            raise HTTPException(status_code=502, detail=f"Provider error: {resp.text[:200]}")

        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0].get("text", "")
        except Exception:
            raise HTTPException(status_code=502, detail="Malformed provider response.")
        if not text:
            raise HTTPException(status_code=502, detail="Empty provider response.")
        return text

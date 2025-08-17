"""
LLMService (Gemini) — Production-Ready

Covers Milestones 1, 2, and 3 in one place:

M1 (Foundation & API):
  - Loads config from env via pydantic-settings
  - Integrates Google Gemini with timeouts
  - Basic & friendly error handling
  - Clean separation of concerns (service layer)

M2 (Core Chat & Security):
  - Per-session conversation history (optional, opt-in via session_id)
  - Message validation (length, empties) + sanitization
  - Rate limiting (simple token bucket, per session)
  - Request size/length validation + history trimming (budget)
  - Robust error taxonomy: invalid key, timeouts, rate limits, malformed response
  - API key rotation (supports multiple keys via GEMINI_API_KEYS)

M3 (Readiness & Quality):
  - Detailed docstrings & type hints
  - Structured logging (no secrets logged)
  - Extensible design (swap model, add params, replace rate limiter / storage)
  - Returns structured result (ChatResult) for richer telemetry when needed
    while keeping `get_response()` backward-compatible (returns str)

Expected environment variables (.env):
  GEMINI_API_KEY=your_primary_key                 # or
  GEMINI_API_KEYS=key1,key2,key3                  # rotation pool (overrides GEMINI_API_KEY)
  GEMINI_MODEL=gemini-1.5-flash                   # default if unset (set in Settings)

Usage (simple):
  llm = LLMService()
  text = llm.get_response("Hello")  # single-turn
  # or with conversation context:
  text = llm.get_response("Continue...", session_id="user-123")

If you want richer metadata:
  result = llm.chat("Hello", session_id="user-123")
  print(result.reply, result.model, result.usage)

NOTE:
- This in-memory session history is for a single server process. If you run
  multiple workers or want persistence beyond process lifetime, swap the
  in-memory dict for Redis / database storage (interface stays the same).
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Deque, Dict, List, Optional, Tuple

import google.generativeai as genai

from backend.core.config import settings


# ------------------------------------------------------------------------------
# Logging setup (app-wide logger is recommended; here we keep a service-specific one)
# ------------------------------------------------------------------------------
logger = logging.getLogger("llm_service")
if not logger.handlers:
    # Basic sane defaults; in real deployments, prefer a JSON logger / structured logs
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | LLMService | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------------------
# Public data types
# ------------------------------------------------------------------------------
@dataclass
class ChatResult:
    """Rich result object for production telemetry and UI metadata."""
    reply: str
    model: str
    total_history_turns: int
    input_chars: int
    usage: Optional[dict] = None
    latency_ms: Optional[int] = None
    from_cache: bool = False  # reserved for future optimization


# ------------------------------------------------------------------------------
# Internal helpers: rate limiting & error taxonomy
# ------------------------------------------------------------------------------
class TokenBucket:
    """
    Simple token bucket for per-session rate limiting.
    - capacity: max tokens in bucket
    - refill_rate: tokens per second
    - consume(n): returns True if enough tokens; otherwise False
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def consume(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""


class InvalidAPIKeyError(LLMServiceError):
    """Raised when API key is invalid / unauthorized."""


class RateLimitError(LLMServiceError):
    """Raised when rate limit is exceeded."""


class TimeoutError(LLMServiceError):
    """Raised when request times out."""


class MalformedResponseError(LLMServiceError):
    """Raised when response is missing expected data."""


class ValidationError(LLMServiceError):
    """Raised when inputs are invalid (length, empties, etc.)."""


# ------------------------------------------------------------------------------
# Main Service
# ------------------------------------------------------------------------------
class LLMService:
    """
    Production-ready Gemini client wrapper with:
      - input validation & sanitization
      - optional conversation history per session_id
      - rate limiting (per session)
      - retries + exponential backoff for transient errors
      - API key rotation support
      - structured logging
    """

    # ---- Sensible defaults (override via constructor if needed) ----
    DEFAULT_MODEL = settings.gemini_model
    TIMEOUT_S = 20
    MAX_OUTPUT_TOKENS = 512
    TEMPERATURE = 0.4
    TOP_P = 0.95

    # Validation limits (keep UI/responses snappy)
    MAX_MESSAGE_CHARS = 4000
    MAX_HISTORY_TURNS = 16  # total turns kept (user + model); keep small for latency
    MAX_HISTORY_CHARS = 16000  # safety budget for combined history

    # Rate limiting: ~30 messages per minute per session
    RL_CAPACITY = 30
    RL_REFILL_RATE = 0.5  # tokens/sec == 30/min

    # Retry policy
    RETRIES = 3
    BACKOFF_BASE = 0.4  # seconds
    BACKOFF_JITTER = 0.25  # random jitter added

    def __init__(self,
                 model: Optional[str] = None,
                 timeout_s: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 max_output_tokens: Optional[int] = None) -> None:
        # ---- Config & model selection ----
        self.model_name = (model or self.DEFAULT_MODEL).strip()

        self.timeout_s = timeout_s or self.TIMEOUT_S
        self.temperature = self.TEMPERATURE if temperature is None else temperature
        self.top_p = self.TOP_P if top_p is None else top_p
        self.max_output_tokens = max_output_tokens or self.MAX_OUTPUT_TOKENS

        # ---- API keys & rotation pool ----
        # Supports GEMINI_API_KEYS="k1,k2,..." or fallback to GEMINI_API_KEY
        keys_raw = (getattr(settings, "gemini_api_key", None) or "").strip()
        keys_pool_raw = (getattr(settings, "gemini_api_keys", None)
                         if hasattr(settings, "gemini_api_keys") else None)

        if keys_pool_raw:
            # If you prefer multiple keys, add this to your Settings model and .env
            # GEMINI_API_KEYS=key1,key2,key3
            pool = [k.strip() for k in str(keys_pool_raw).split(",") if k.strip()]
        else:
            pool = [k for k in [keys_raw] if k]

        if not pool:
            raise InvalidAPIKeyError("No Gemini API key(s) provided in environment.")

        self._keys: List[str] = pool
        self._key_index = 0
        self._lock = RLock()  # guard key rotation & shared maps

        # ---- In-memory per-session state ----
        # Map session_id -> deque of {"role": "user"|"model", "content": "..."}
        self._history: Dict[str, Deque[Dict[str, str]]] = {}
        # Map session_id -> TokenBucket
        self._buckets: Dict[str, TokenBucket] = {}

        # ---- Initialize SDK once with the first key ----
        with self._lock:
            genai.configure(api_key=self._current_key())
            self._model = self._make_model()

        logger.info("LLMService initialized (model=%s, keys=%d)",
                    self.model_name, len(self._keys))

    # -------------------------- Public API --------------------------

    def get_response(self,
                     message: str,
                     *,
                     session_id: Optional[str] = None,
                     system_instruction: Optional[str] = None) -> str:
        """
        Backward-compatible method that returns only the reply text and
        never raises errors — converts them to friendly strings.

        For richer metadata and explicit error handling, use `chat(...)` instead.
        """
        try:
            result = self.chat(
                message=message,
                session_id=session_id,
                system_instruction=system_instruction
            )
            return result.reply
        except ValidationError as e:
            logger.warning("Validation error: %s", e)
            return str(e)  # user-friendly message already
        except InvalidAPIKeyError:
            return "Authentication failed: invalid API key. Please check your configuration."
        except RateLimitError:
            return "I’m receiving too many requests right now. Please wait a moment and try again."
        except TimeoutError:
            return "The request to the AI model timed out. Please retry."
        except MalformedResponseError:
            return "I couldn’t understand the AI response. Please try again."
        except Exception as e:
            logger.exception("Unexpected error in get_response: %s", e)
            return "Something went wrong while generating a response."

    def chat(self,
             message: str,
             *,
             session_id: Optional[str] = None,
             system_instruction: Optional[str] = None,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             max_output_tokens: Optional[int] = None) -> ChatResult:
        """
        Production-grade entrypoint.
        - Validates & sanitizes input
        - Applies per-session rate limiting
        - Adds conversation context when `session_id` is provided
        - Retries transient failures with exponential backoff
        - Rotates API keys on auth/rate-limit when possible
        - Returns a rich ChatResult
        - May raise typed LLMServiceError subclasses for fine-grained handling
        """
        # 1) Validate & sanitize
        user_text = self._sanitize(self._validate_message(message))

        # 2) Rate limit (only if we can identify a session)
        if session_id is not None:
            if not self._check_rate_limit(session_id):
                raise RateLimitError("Too many requests. Slow down and try again.")
        # 3) Build contents with (optional) history
        contents = self._build_contents(user_text, session_id=session_id)

        # 4) Generation options
        gen_config = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_output_tokens": self.max_output_tokens if max_output_tokens is None else max_output_tokens,
        }

        # 5) Call model with retries + key rotation
        start = time.time()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.RETRIES + 1):
            try:
                # If system_instruction is provided, build a temporary model with it
                model = self._model
                if system_instruction:
                    model = genai.GenerativeModel(self.model_name, system_instruction=system_instruction)

                # Gemini call
                resp = model.generate_content(
                    contents=contents,
                    generation_config=gen_config,
                    request_options={"timeout": self.timeout_s},
                )

                reply = (getattr(resp, "text", None) or "").strip()
                if not reply:
                    raise MalformedResponseError("No text returned by the model.")

                latency_ms = int((time.time() - start) * 1000)

                # 6) Update history after successful call
                total_turns = self._update_history_after_success(
                    session_id=session_id, user_text=user_text, reply_text=reply
                )

                # Try to capture basic usage metadata if available
                usage = None
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um:
                        usage = {
                            "input_characters": getattr(um, "input_characters", None),
                            "output_characters": getattr(um, "output_characters", None),
                            "total_characters": getattr(um, "total_characters", None),
                        }
                except Exception:
                    # Best-effort; ignore if SDK fields change
                    usage = None

                logger.info("Gemini call success (session=%s, latency_ms=%s, attempt=%d)",
                            session_id, latency_ms, attempt)

                return ChatResult(
                    reply=reply,
                    model=self.model_name,
                    total_history_turns=total_turns,
                    input_chars=len(user_text),
                    usage=usage,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_exc = e
                # Categorize and decide if we should retry or rotate key
                action, classification = self._classify_error(e)
                logger.warning("Gemini call error (attempt=%d/%d, class=%s, action=%s, session=%s): %s",
                               attempt, self.RETRIES, classification, action, session_id, e)

                if action == "rotate_key":
                    # Attempt key rotation if pool > 1
                    if not self._rotate_key():
                        # no other keys available — fall back to retry/backoff if marked transient
                        pass
                    else:
                        # reconfigure the SDK with the new key
                        with self._lock:
                            genai.configure(api_key=self._current_key())
                            self._model = self._make_model()

                if action in ("retry", "rotate_key_retry"):
                    # exponential backoff with jitter
                    sleep_s = self._backoff_time(attempt)
                    time.sleep(sleep_s)
                    continue

                # Do not retry for non-retryable errors
                break

        # If we reach here, all attempts failed — raise the most appropriate error
        assert last_exc is not None
        _, classification = self._classify_error(last_exc)

        if classification == "invalid_key":
            raise InvalidAPIKeyError("Invalid or unauthorized API key.")
        if classification == "timeout":
            raise TimeoutError("The model request timed out.")
        if classification == "rate_limit":
            raise RateLimitError("Rate limited by API.")
        if classification == "malformed":
            raise MalformedResponseError("Malformed response from model.")
        if classification == "validation":
            raise ValidationError(str(last_exc))

        # Unknown/other
        raise LLMServiceError(str(last_exc))

    # ----------------------- Session & History API -----------------------

    def clear_session(self, session_id: str) -> None:
        """Remove conversation history and rate limiter state for a session."""
        with self._lock:
            self._history.pop(session_id, None)
            self._buckets.pop(session_id, None)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return a copy of the current history for inspection/testing."""
        with self._lock:
            dq = self._history.get(session_id, deque())
            return list(dq)

    # ----------------------- Internal helpers -----------------------

    def _validate_message(self, message: str) -> str:
        """Basic validation for empty/oversized messages."""
        if message is None:
            raise ValidationError("Message cannot be null.")
        msg = str(message).strip()
        if not msg:
            raise ValidationError("Please enter a message.")
        if len(msg) > self.MAX_MESSAGE_CHARS:
            raise ValidationError(f"Message exceeds {self.MAX_MESSAGE_CHARS} characters.")
        return msg

    @staticmethod
    def _sanitize(text: str) -> str:
        """
        Remove control chars that can cause logging/transport issues.
        (UI XSS is usually handled by the front-end renderer, but stripping
        non-printable ASCII here is still good hygiene.)
        """
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)

    def _check_rate_limit(self, session_id: str) -> bool:
        """Consume 1 token; return False if over the limit."""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                bucket = TokenBucket(self.RL_CAPACITY, self.RL_REFILL_RATE)
                self._buckets[session_id] = bucket
        return bucket.consume(1.0)

    def _build_contents(self, user_text: str, *, session_id: Optional[str]) -> List[Dict[str, object]]:
        """
        Build Gemini 'contents' array, optionally including past turns.
        Gemini expects roles 'user' and 'model'.
        """
        contents: List[Dict[str, object]] = []

        if session_id is not None:
            with self._lock:
                dq = self._history.get(session_id)
                if dq:
                    # Copy current history (already trimmed by _update_history_after_success)
                    for m in dq:
                        contents.append({"role": m["role"], "parts": [m["content"]]})

        # Append the new user message
        contents.append({"role": "user", "parts": [user_text]})

        # Final safety budget: if total chars exceed MAX_HISTORY_CHARS, drop from the front.
        total_chars = sum(len("".join(msg.get("parts", []))) for msg in contents)
        if total_chars > self.MAX_HISTORY_CHARS:
            # Trim oldest history messages (not the last user message)
            i = 0
            while i < len(contents) - 1 and total_chars > self.MAX_HISTORY_CHARS:
                # compute length of ith message
                msg_len = len("".join(contents[i].get("parts", [])))
                total_chars -= msg_len
                i += 1
            contents = contents[i:]  # drop oldest i messages

        return contents

    def _update_history_after_success(self,
                                      *,
                                      session_id: Optional[str],
                                      user_text: str,
                                      reply_text: str) -> int:
        """
        After a successful generation, append (user, model) to session history
        and enforce max turns + budgets.
        """
        if session_id is None:
            return 0

        with self._lock:
            dq = self._history.get(session_id)
            if dq is None:
                dq = deque()
                self._history[session_id] = dq

            # Append user and model turns
            dq.append({"role": "user", "content": user_text})
            dq.append({"role": "model", "content": reply_text})

            # Trim by turns
            while len(dq) > self.MAX_HISTORY_TURNS:
                dq.popleft()

            # Trim by characters (safety)
            def total_chars() -> int:
                return sum(len(m["content"]) for m in dq)

            while total_chars() > self.MAX_HISTORY_CHARS and dq:
                dq.popleft()

            return len(dq)

    def _current_key(self) -> str:
        return self._keys[self._key_index]

    def _rotate_key(self) -> bool:
        """Rotate to the next API key if available; return True if changed."""
        with self._lock:
            if len(self._keys) <= 1:
                return False
            self._key_index = (self._key_index + 1) % len(self._keys)
            logger.info("Rotated Gemini API key (index=%d)", self._key_index)
            return True

    def _make_model(self):
        """Create a GenerativeModel instance for the current model name."""
        return genai.GenerativeModel(self.model_name)

    def _backoff_time(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        base = self.BACKOFF_BASE * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.BACKOFF_JITTER)
        return base + jitter

    def _classify_error(self, exc: Exception) -> Tuple[str, str]:
        """
        Map exceptions to (action, classification):

        action:
          - 'retry'              -> transient network/timeout
          - 'rotate_key_retry'   -> likely key-related or rate-limit; try rotating & retry
          - 'fail'               -> do not retry

        classification:
          - 'invalid_key' | 'rate_limit' | 'timeout' | 'malformed' | 'validation' | 'other'
        """
        msg = str(exc).lower()

        # NOTE: The google SDK error classes may vary by version.
        # We conservatively inspect message text to avoid hard dependencies.
        if "unauthorized" in msg or "401" in msg or "invalid api key" in msg or "permission denied" in msg:
            return "rotate_key_retry", "invalid_key"

        if "429" in msg or "rate limit" in msg or "resource exhausted" in msg or "quota" in msg:
            return "rotate_key_retry", "rate_limit"

        if "deadline exceeded" in msg or "timeout" in msg:
            return "retry", "timeout"

        if "invalid argument" in msg or "bad request" in msg:
            return "fail", "validation"

        if isinstance(exc, MalformedResponseError):
            return "retry", "malformed"

        # Unknown — treat as non-retryable unless it smells transient
        if "temporarily unavailable" in msg or "try again" in msg or "reset" in msg:
            return "retry", "other"

        return "fail", "other"

from fastapi import HTTPException

class ChatAPIException(HTTPException):
    """Base exception for chat API errors."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)

# Registry mapping status codes to (exception class, default message)
EXCEPTION_REGISTRY = {
    429: (ChatAPIException, "Rate limit from provider. Please retry shortly."),
    503: (ChatAPIException, "Provider overloaded/unavailable."),
    504: (ChatAPIException, "Upstream timeout."),
    502: (ChatAPIException, "Transient upstream error after retries."),
}

def raise_for_status(status_code: int, detail: str = None):
    exc_class, default_msg = EXCEPTION_REGISTRY.get(status_code, (ChatAPIException, "Unknown error"))
    raise exc_class(status_code=status_code, detail=detail or default_msg)
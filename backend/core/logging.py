import logging
import sys
import json
from typing import Any

class JsonFormatter(logging.Formatter):
    """
    Logging formatter to output logs as JSON for console/uvicorn compatibility.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger to output JSON logs to stdout (uvicorn-compatible).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

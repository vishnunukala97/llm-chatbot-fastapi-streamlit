from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

def get_backend_url() -> str:
    """Get backend URL from env or default."""
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def get_chat_data_dir() -> Path:
    """Get chat data directory from env or default, ensure it exists."""
    dir_path = Path(os.getenv("CHAT_DATA_DIR", "./data/chats/"))
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_max_history() -> int:
    """Get max chat history to display from env or default."""
    try:
        return int(os.getenv("CHAT_MAX_HISTORY", "30"))
    except Exception:
        return 30

def get_max_prompt_len() -> int:
    """Get max prompt length from env or default."""
    try:
        return int(os.getenv("CHAT_MAX_PROMPT_LEN", "5000"))
    except Exception:
        return 5000
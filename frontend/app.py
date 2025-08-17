"""
Streamlit UI:

- Keeps a stable session_id (UUID) per browser session.
- Shows chat history with user/assistant roles.
- Calls FastAPI /chat with X-Session-Id header so the backend can keep context.
- Adds 'Clear chat' that also clears server-side history via /clear-session.
- Displays latency and model metadata (M3 UX polish).

Run:
1) uvicorn backend.main:app --reload
2) streamlit run frontend/app.py
"""

import uuid
import time
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Basic LLM Chatbot (Gemini)", page_icon="💬")
st.title("💬 Basic LLM Chatbot (Gemini)")

# 1) Ensure a stable session id per browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 2) Initialize local UI history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Controls (Clear chat)
col_clear, col_spacer = st.columns([1, 9])
if col_clear.button("Clear chat"):
    try:
        requests.post(
            f"{BACKEND_URL}/clear-session",
            headers={"X-Session-Id": st.session_state.session_id},
            timeout=10,
        )
    except Exception:
        pass  # ignore network issues during clear
    st.session_state.messages = []
    st.experimental_rerun()

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    # 1) Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Ask backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            try:
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt},
                    headers={"X-Session-Id": st.session_state.session_id},
                    timeout=20,
                )
                data = r.json()
                reply = data.get("reply", "Error: No reply")
                model = data.get("model") or "unknown-model"
                latency_ms = data.get("latency_ms")
                meta = f"\n\n<span style='opacity:0.6;font-size:0.85em'>⏱️ {latency_ms} ms · {model}</span>"
            except Exception as e:
                reply = f"Error calling backend: {e}"
                meta = ""
            st.markdown(reply + meta, unsafe_allow_html=True)

    # 3) Save assistant message to local UI history
    st.session_state.messages.append({"role": "assistant", "content": reply})

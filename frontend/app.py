import streamlit as st
import requests
import time
import json
from config import (
    get_backend_url,
    get_chat_data_dir,
    get_max_history,
    get_max_prompt_len,
)
from utils.storage import (
    ensure_data_dir,
    list_chats,
    load_chat,
    save_chat,
    new_chat,
    safe_filename,
    sanitize_input,
)

BACKEND_URL = get_backend_url()
CHAT_DATA_DIR = get_chat_data_dir()
MAX_HISTORY = get_max_history()
MAX_PROMPT_LEN = get_max_prompt_len()

st.set_page_config(page_title="Basic LLM Chatbot", page_icon="💬", layout="centered")
st.title("💬 Basic LLM Chatbot")

# --- Session state management ---
if "current_chat_id" not in st.session_state:
    chats, _ = list_chats()
    if chats:
        st.session_state.current_chat_id = chats[0]["id"]
    else:
        chat = new_chat()
        save_chat(chat)
        st.session_state.current_chat_id = chat["id"]

def load_current_chat():
    chat = load_chat(st.session_state.current_chat_id)
    if chat is None:
        chat = new_chat()
        save_chat(chat)
        st.session_state.current_chat_id = chat["id"]
    st.session_state.messages = chat["messages"].copy()
    st.session_state.current_chat = chat

if "messages" not in st.session_state or "current_chat" not in st.session_state:
    load_current_chat()

# --- Sidebar: Chat management ---
with st.sidebar:
    chats, warnings = list_chats()
    chat_options = [(c["id"], c["title"]) for c in chats]
    chat_id_to_title = {c["id"]: c["title"] for c in chats}
    current_idx = 0
    if st.session_state.current_chat_id in chat_id_to_title:
        current_idx = [i for i, (cid, _) in enumerate(chat_options) if cid == st.session_state.current_chat_id][0]
    if chat_options:
        selected = st.selectbox(
            "Chats",
            options=chat_options,
            format_func=lambda x: x[1],
            index=current_idx,
            key="chat_selectbox"
        )
        if selected[0] != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected[0]
            load_current_chat()
            st.rerun()
    if st.button("New chat"):
        chat = new_chat()
        save_chat(chat)
        st.session_state.current_chat_id = chat["id"]
        load_current_chat()
        st.rerun()
    if st.button("Delete chat"):
        if st.session_state.current_chat_id:
            f = CHAT_DATA_DIR / f"{st.session_state.current_chat_id}.json"
            if f.exists():
                f.unlink()
            chats, _ = list_chats()
            if chats:
                st.session_state.current_chat_id = chats[0]["id"]
            else:
                chat = new_chat()
                save_chat(chat)
                st.session_state.current_chat_id = chat["id"]
            load_current_chat()
            st.rerun()
    new_title = st.text_input("Rename chat", value=st.session_state.current_chat["title"], key="rename_input")
    if st.button("Save name"):
        st.session_state.current_chat["title"] = new_title.strip() or "Untitled"
        save_chat(st.session_state.current_chat)
        st.rerun()
    chat_json = json.dumps(st.session_state.current_chat, indent=2, ensure_ascii=False)
    st.download_button("Download chat", chat_json, file_name=f"{safe_filename(st.session_state.current_chat['title'])}.json", mime="application/json")
    for w in warnings:
        st.warning(w)

# --- Main chat UI ---
for m in st.session_state.messages[-MAX_HISTORY:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Type your message…")
if user_msg:
    user_msg = sanitize_input(user_msg, MAX_PROMPT_LEN)
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.current_chat["messages"] = st.session_state.messages
    save_chat(st.session_state.current_chat)
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                t0 = time.perf_counter()
                r = requests.post(f"{BACKEND_URL}/chat", json={"message": user_msg}, timeout=30)
                elapsed = int((time.perf_counter() - t0) * 1000)
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "")
                    st.markdown(reply or "_(empty response)_")
                    st.caption(f"⏱️ {data.get('latency_ms', elapsed)} ms")
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                elif r.status_code in (401, 403):
                    st.error("Auth error: check your API key.")
                    st.session_state.messages.append({"role": "assistant", "content": "Auth error."})
                elif r.status_code == 429:
                    st.error("Rate limit. Please wait a moment and retry.")
                    st.session_state.messages.append({"role": "assistant", "content": "Rate limit; retry soon."})
                else:
                    st.error(f"Server error: {r.text[:200]}")
                    st.session_state.messages.append({"role": "assistant", "content": "Server error."})
            except requests.Timeout:
                st.error("Timeout—please try again.")
                st.session_state.messages.append({"role": "assistant", "content": "Timeout—please try again."})
            except Exception as e:
                st.error("Unexpected error.")
                st.session_state.messages.append({"role": "assistant", "content": "Unexpected error."})
    st.session_state.current_chat["messages"] = st.session_state.messages
    save_chat(st.session_state.current_chat)

st.divider()
if st.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.current_chat["messages"] = []
    save_chat(st.session_state.current_chat)
    st.rerun()

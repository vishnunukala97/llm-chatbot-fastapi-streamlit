# basic-llm-chatbot

A Python 3.11 project scaffold for a basic LLM chatbot with FastAPI backend and Streamlit frontend.

## Environment Setup

1. Copy `.env.example` to `.env` and fill in your provider keys and settings:

   ```sh
   cp .env.example .env
   # or manually copy and edit the file
   ```

2. The `.env` file should NOT be committed to version control. It is already excluded by `.gitignore`.

3. The app uses [python-dotenv](https://pypi.org/project/python-dotenv/) to load environment variables automatically.

## Install & Run

1. Install dependencies (in a virtual environment):
   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy and configure your environment variables:
   ```sh
   cp .env.example .env
   # Edit .env as needed
   ```

3. Run the backend:
   ```sh
   uvicorn backend.main:app --reload
   ```

   - The API will be available at: http://localhost:8000
   - Health check: http://localhost:8000/healthz
   - Chat endpoint: http://localhost:8000/api/chat

4. Run the frontend:
   ```sh
   streamlit run frontend/app.py
   ```

# LLM Chatbot (FastAPI + Streamlit)

## Prerequisites
- Python 3.10+
- [OpenAI API key](https://platform.openai.com/) and/or [Google Gemini API key](https://ai.google.dev/gemini-api/docs/get-started)

## Setup
```sh
# Clone the repo and cd into the project root
python -m venv .venv
.venv\Scripts\activate  # On Windows
# Or: source .venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

## Configuring Environment Variables
- Copy `.env.example` to `.env` and fill in your API keys:
  ```
  PROVIDER=openai
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=...
  BACKEND_HOST=127.0.0.1
  BACKEND_PORT=8000
  OPENAI_MODEL=gpt-4o-mini
  TIMEOUT_SECONDS=30
  RATE_LIMIT_QPS=2
  ```

## Running the Backend (API)
```sh
uvicorn backend.main:app --reload
```
- The API will be available at: http://localhost:8000
- Health check: http://localhost:8000/healthz
- Chat endpoint: http://localhost:8000/api/chat

## Running the Frontend (UI)
```sh
streamlit run frontend/app.py
```
- The UI will open at: http://localhost:8501

## Quick API Example (curl)
```sh
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "provider": "openai"}'
```

## Basic Troubleshooting
- **Missing API key**: Ensure `.env` exists and contains your real API keys.
- **Rate limit errors**: Wait and try again, or upgrade your API plan.
- **Gemini 404 errors**: Double-check your Gemini API key and that the API is enabled for your Google Cloud project.
- **Module not found**: Activate your virtual environment and install requirements.
- **CORS errors**: Make sure backend is running and accessible at http://localhost:8000.

For more help, check logs in your terminal or ask for support!

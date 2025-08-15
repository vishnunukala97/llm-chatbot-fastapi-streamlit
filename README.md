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

4. Run the frontend:
   ```sh
   streamlit run frontend/app.py
   ```

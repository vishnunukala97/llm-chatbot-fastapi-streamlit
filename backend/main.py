from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import chat

app = FastAPI()

# Allow Streamlit frontend
origins = ["http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Include chat router under /api
app.include_router(chat.router, prefix="/api")

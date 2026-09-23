from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_pipeline import answer_question


app = FastAPI(
    title="AI Customer Support Chatbot",
    description="RAG-based customer support chatbot",
    version="1.0"
)


# Allow the frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@app.get("/")
def home():
    return {
        "message": "AI Customer Support Chatbot API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = answer_question(request.message)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
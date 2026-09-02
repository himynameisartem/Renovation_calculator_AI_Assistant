from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import LONG_QUESTION_MESSAGE, MAX_QUESTION_CHARS, RenovationRAG


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources_count: int


app = FastAPI(title="Renovation RAG API")


@lru_cache(maxsize=1)
def get_rag() -> RenovationRAG:
    return RenovationRAG()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = " ".join(request.message.split()).strip()

    if len(message) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=LONG_QUESTION_MESSAGE)

    result = get_rag().answer(message)

    return ChatResponse(
        answer=result.answer,
        intent=result.intent,
        sources_count=len(result.points),
    )

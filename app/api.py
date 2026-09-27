from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import LONG_QUESTION_MESSAGE, MAX_QUESTION_CHARS, RenovationRAG
from app.photo_estimate import PhotoEstimateService

MAX_HISTORY_MESSAGES = 10


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    messages: list[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES + 1,
    )


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources_count: int


class DetectedMaterial(BaseModel):
    material: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class SurfaceAnalysis(BaseModel):
    walls: list[DetectedMaterial] = Field(default_factory=list)
    floor: list[DetectedMaterial] = Field(default_factory=list)
    ceiling: list[DetectedMaterial] = Field(default_factory=list)


class PhotoEstimateRequest(BaseModel):
    room_type: Literal["living", "kitchen", "bathroom", "hallway", "other"] = "other"
    room_name: str = Field(default="", max_length=100)
    area_m2: float = Field(gt=0.0, le=1000.0)
    height_m: float = Field(gt=0.0, le=20.0)
    surfaces: SurfaceAnalysis


class PhotoEstimateResponse(BaseModel):
    answer: str
    sources_count: int


app = FastAPI(title="Renovation RAG API")


@lru_cache(maxsize=1)
def get_rag() -> RenovationRAG:
    return RenovationRAG()


@lru_cache(maxsize=1)
def get_photo_estimate_service() -> PhotoEstimateService:
    return PhotoEstimateService(rag=get_rag())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    history: list[dict[str, str]] = []

    if request.messages:
        messages = request.messages[-(MAX_HISTORY_MESSAGES + 1):]
        if messages[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="Последнее сообщение должно быть от пользователя.",
            )

        message = " ".join(messages[-1].content.split()).strip()
        history = [
            {
                "role": item.role,
                "content": " ".join(item.content.split()).strip(),
            }
            for item in messages[:-1]
        ]
    else:
        message = " ".join((request.message or "").split()).strip()

    if not message:
        raise HTTPException(status_code=400, detail="Сообщение не должно быть пустым.")

    if len(message) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=LONG_QUESTION_MESSAGE)

    result = get_rag().answer(message, history=history)

    return ChatResponse(
        answer=result.answer,
        intent=result.intent,
        sources_count=len(result.points),
    )


@app.post("/estimate/from-cv", response_model=PhotoEstimateResponse)
def estimate_from_cv(request: PhotoEstimateRequest) -> PhotoEstimateResponse:
    try:
        result = get_photo_estimate_service().answer(
            cv_result=request.surfaces.model_dump(),
            room_type=request.room_type,
            room_name=request.room_name,
            area_m2=request.area_m2,
            height_m=request.height_m,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PhotoEstimateResponse(
        answer=result.answer,
        sources_count=len(result.points),
    )

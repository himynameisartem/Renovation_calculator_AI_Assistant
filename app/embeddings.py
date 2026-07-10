from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

@dataclass(slots=True)
class EmbeddedChunk:
    text: str
    metadata: dict
    embedding: list[float]


class OpenAIEmbeddingClient:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            embeddings.extend(item.embedding for item in response.data)

        return embeddings
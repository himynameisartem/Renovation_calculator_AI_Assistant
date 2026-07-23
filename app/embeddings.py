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
        base_url: str | None = None,
        project: str | None = None,
        encoding_format: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.project = project
        self.encoding_format = encoding_format

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            project=self.project,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            input_value = batch[0] if self.batch_size == 1 else batch

            kwargs = {
                "model": self.model,
                "input": input_value,
            }

            if self.encoding_format:
                kwargs["encoding_format"] = self.encoding_format

            response = self.client.embeddings.create(**kwargs)

            embeddings.extend(item.embedding for item in response.data)

        return embeddings

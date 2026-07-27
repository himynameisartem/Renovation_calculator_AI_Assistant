from __future__ import annotations

import hashlib
import os
import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.embeddings import EmbeddedChunk


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        qdrant_url = url or os.getenv("QDRANT_URL")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            self.client = QdrantClient(host=host, port=port)

    def recreate_collection(self, vector_size: int) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upload_chunks(
        self,
        chunks: Iterable[EmbeddedChunk],
        batch_size: int = 100,
    ) -> None:
        batch: list[EmbeddedChunk] = []

        for chunk in chunks:
            batch.append(chunk)

            if len(batch) >= batch_size:
                self._upload_batch(batch)
                batch = []

        if batch:
            self._upload_batch(batch)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        query_filter: models.Filter | None = None,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _upload_batch(self, chunks: list[EmbeddedChunk]) -> None:
        points: list[models.PointStruct] = []

        for chunk in chunks:
            payload = dict(chunk.metadata or {})
            payload["text"] = chunk.text

            points.append(
                models.PointStruct(
                    id=self._build_point_id(chunk),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _build_point_id(self, chunk: EmbeddedChunk) -> str:
        url = str(chunk.metadata.get("url", ""))
        chunk_index = str(chunk.metadata.get("chunk_index", ""))
        text = chunk.text or ""

        raw = f"{url}\n{chunk_index}\n{text}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

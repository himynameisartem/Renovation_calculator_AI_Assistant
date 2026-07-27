from __future__ import annotations

import os
from pathlib import Path
import pickle

from dotenv import load_dotenv

from app.embeddings import EmbeddedChunk
from app.vector_store import QdrantVectorStore

load_dotenv()


INPUT_PATH = Path("data/cleaned/embedded_chunks.pkl")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "renovation_docs")


def load_embedded_chunks(input_path: Path) -> list[EmbeddedChunk]:
    with input_path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    embedded_chunks = load_embedded_chunks(INPUT_PATH)

    if not embedded_chunks:
        raise ValueError("No embedded chunks found")

    vector_size = len(embedded_chunks[0].embedding)

    store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
    )

    store.recreate_collection(vector_size=vector_size)
    store.upload_chunks(embedded_chunks, batch_size=100)

    print(f"embedded_chunks loaded: {len(embedded_chunks)}")
    print(f"vector_size: {vector_size}")
    print(f"collection: {COLLECTION_NAME}")
    print(f"qdrant_url: {os.getenv('QDRANT_URL') or 'localhost:6333'}")
    print("upload complete")


if __name__ == "__main__":
    main()

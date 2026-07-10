from __future__ import annotations

from pathlib import Path
import pickle

from app.embeddings import EmbeddedChunk
from app.vector_store import QdrantVectorStore


INPUT_PATH = Path("data/cleaned/embedded_chunks.pkl")
COLLECTION_NAME = "renovation_docs"


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
        host="localhost",
        port=6333,
    )

    store.recreate_collection(vector_size=vector_size)
    store.upload_chunks(embedded_chunks, batch_size=100)

    print(f"embedded_chunks loaded: {len(embedded_chunks)}")
    print(f"vector_size: {vector_size}")
    print(f"collection: {COLLECTION_NAME}")
    print("upload complete")


if __name__ == "__main__":
    main()
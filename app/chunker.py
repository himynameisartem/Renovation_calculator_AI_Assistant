from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in documents:
            text = (doc.text or "").strip()
            if not text:
                continue

            parts = self.splitter.split_text(text)
            total_parts = len(parts)

            for index, part in enumerate(parts):
                chunk_text = part.strip()
                if not chunk_text:
                    continue

                metadata = dict(doc.metadata or {})
                metadata["chunk_index"] = index
                metadata["chunk_total"] = total_parts

                chunks.append(
                    Document(
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

        return chunks
from pathlib import Path
import pickle
import tiktoken

with Path("data/cleaned/chunks.pkl").open("rb") as f:
    chunks = pickle.load(f)

encoding = tiktoken.get_encoding("cl100k_base")

total_tokens = 0

for chunk in chunks:
    text = chunk.text or ""
    total_tokens += len(encoding.encode(text))

print(f"chunks: {len(chunks)}")
print(f"total_tokens: {total_tokens}")
print(f"avg_tokens_per_chunk: {total_tokens / len(chunks):.2f}")
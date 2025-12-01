import os
import json
import faiss
import numpy as np
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ---------------- CONFIGURE HUGGINGFACE TOKEN ----------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") # ensure this is set in your environment

client = InferenceClient(token=HF_TOKEN)


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(texts):
    embeddings = []
    for text in texts:
        response = client.feature_extraction(
            text,  # <-- FIXED: positional argument
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings.append(response)
    return np.array(embeddings).astype("float32")


def build_faiss_index(chunks, index_path, metadata_path):
    texts = [c["content"] for c in chunks]

    print("Embedding text chunks... (This may take some time)")
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print("\nFAISS index built successfully.")
    print(f"Total vectors stored: {index.ntotal}")


if __name__ == "__main__":
    chunks = load_chunks("data/processed/combined_chunks.jsonl")
    os.makedirs("data/processed", exist_ok=True)

    build_faiss_index(
        chunks,
        index_path="data/processed/faiss_index.bin",
        metadata_path="data/processed/chunk_metadata.json"
    )

import os
import faiss
import json
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

# Load FAISS index
index = faiss.read_index("data/processed/faiss_index.bin")

with open("data/processed/chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


def embed_query(text):
    response = client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(response).astype("float32")


def search_faiss(query, k=5):
    query_vector = embed_query(query)
    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]


def call_hf_llm(prompt):
    response = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",  # free model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return response.choices[0].message["content"]


def generate_answer(query):
    refs = search_faiss(query)

    context = "\n\n".join([f"[page {r.get('page')}] {r['content']}" for r in refs])

    prompt = f"""
Answer the question ONLY based on CONTEXT.
Cite page numbers exactly like [page X]. 
If answer is not in context, say "Not found in the document."

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

    answer_text = call_hf_llm(prompt)
    return answer_text, refs


if __name__ == "__main__":
    q = input("Ask a question: ")
    ans, refs = generate_answer(q)
    print("\nANSWER:\n", ans)
    print("\nREFERENCES PAGES:\n", [r.get("page") for r in refs])

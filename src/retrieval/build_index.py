import os
import json
import faiss
import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

# -------- LOAD ENV VARIABLES --------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -------- LOAD EMBEDDING CLIENT --------
client = InferenceClient(token=HF_TOKEN)

# -------- LOAD FAISS INDEX --------
index = faiss.read_index("data/processed/faiss_index.bin")
with open("data/processed/chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -------- LOCAL GENERATION MODEL --------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# -------- EMBEDDING FUNCTION --------
def embed_query(text):
    response = client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(response).astype("float32")


# -------- SEARCH RETRIEVAL --------
def search_faiss(query, k=5):
    query_vector = embed_query(query)
    query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = index.search(query_vector, k)

    results = []
    pages = []
    for idx in indices[0]:
        results.append(metadata[idx])
        pages.append(metadata[idx].get("page"))

    return results, pages


# -------- GENERATE RAG ANSWER --------
def generate_answer(query):
    retrieved, pages = search_faiss(query)

    context = "\n\n".join(
        [f"[page {r.get('page')}] {r['content']}" for r in retrieved]
    )

    prompt = (
        "Use the following CONTEXT to answer the QUESTION. "
        "Only use facts from the context and cite page numbers.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, pages


# -------- CLI MAIN --------
if __name__ == "__main__":
    q = input("Ask a question: ")
    ans, pages = generate_answer(q)

    print("\nANSWER:\n", ans)
    print("\nREFERENCES (pages):", pages)

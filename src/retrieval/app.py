import streamlit as st
import faiss
import numpy as np
import json
from huggingface_hub import InferenceClient
import os

# Load keys securely
HF_TOKEN = os.getenv("HF_TOKEN")  # Streamlit cloud will store this secretly
client = InferenceClient(token=HF_TOKEN)

# Load FAISS index and metadata
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
    pages = [metadata[i]["page"] for i in indices[0]]
    return pages


# ------------------ STREAMLIT UI -------------------
st.title("📘 Qatar IMF Report RAG Search Engine")
st.write("Ask anything about Qatar’s economic analysis based on IMF report")

query = st.text_input("Enter your question")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please type a query")
    else:
        pages = search_faiss(query)
        st.subheader("📍 Source References")
        st.write("Pages Retrieved:", pages)

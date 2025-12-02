import streamlit as st
import faiss
import numpy as np
import json
from huggingface_hub import InferenceClient
import os

# Load secure token from Streamlit Secrets or .env locally
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN missing. Add it to Streamlit Secrets.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

# Load FAISS index and metadata
index = faiss.read_index("data/processed/faiss_index.bin")
with open("data/processed/chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def embed_query(text):
    try:
        response = client.feature_extraction(
            text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        return np.array(response).astype("float32")
    except Exception as e:
        return None

def search_faiss(query, k=5):
    query_vector = embed_query(query)
    if query_vector is None:
        return None

    query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)

    pages = []
    for i in indices[0]:
        p = metadata[i].get("page")
        if p in [None, "", "null"]:
            continue
        pages.append(p)
    return pages


# ---------------- STREAMLIT UI ----------------
st.title("📘 Qatar IMF Report - RAG Search Engine")
st.write("Ask anything about Qatar’s Article IV IMF economic report. Only context retrieval is performed — no LLM answering yet.")

query = st.text_input("🔍 Enter your question")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠ Please type a question")
    else:
        with st.spinner("Searching relevant document pages..."):
            pages = search_faiss(query)

        st.subheader("📍 Retrieved Reference Pages")

        if pages is None:
            st.error("❌ Embedding API failed. Check HF Token / Rate limits.")
        elif len(pages) == 0:
            st.write("No relevant pages found for this query.")
        else:
            st.success(f"Found references on pages: {pages}")

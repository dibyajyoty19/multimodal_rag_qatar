import streamlit as st
from src.retrieval.query_engine import generate_answer

st.set_page_config(page_title="Multi-Modal RAG Qatar", layout="wide")

st.title("📄 Multi-Modal RAG Chatbot – Qatar IMF Report")
st.write("Ask any question related to the uploaded IMF Article IV document.")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching document and generating answer..."):
        answer, pages = generate_answer(query)

    st.subheader("📌 Answer")
    st.write(answer)

    st.subheader("📎 Page References")
    st.write(pages)

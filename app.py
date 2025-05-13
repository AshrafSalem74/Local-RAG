import streamlit as st
from rag import load_or_create_vectorstore, update_vectorstore, get_relevant_chunks
from llm import generate_answer
from file_loader import extract_text_from_file
#test
st.set_page_config(page_title="🧠 Local RAG", layout="wide")
st.title("🔐 Local RAG with Dynamic File Support")

vectorstore = load_or_create_vectorstore()

uploaded_file = st.file_uploader("📁 Upload a file (.txt, .pdf, .docx, .md, etc.)", type=["txt", "pdf", "docx", "md", "html", "csv"])
if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    if text.startswith("❌"):
        st.error(text)
    else:
        vectorstore = update_vectorstore(vectorstore, text)
        st.success("✅ Vector store updated with new content.")

query = st.text_input("💬 Ask a question:")
if query:
    with st.spinner("🔍 Thinking..."):
        relevant_docs = get_relevant_chunks(vectorstore, query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        answer = generate_answer(context, query)
        st.markdown("### 💡 Answer:")
        st.success(answer)

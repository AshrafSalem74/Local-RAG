import os
import pickle
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
INDEX_FILE = "vector_store.pkl"

def load_or_create_vectorstore():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return FAISS.from_documents([], embedding_model, embedding_function=lambda x: embedding_model.encode([x])[0])

def update_vectorstore(faiss_index, new_text):
    documents = splitter.split_documents([Document(page_content=new_text)])
    faiss_index.add_documents(documents)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(faiss_index, f)
    return faiss_index

def get_relevant_chunks(faiss_index, query):
    return faiss_index.similarity_search(query, k=3)

import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
INDEX_FILE = "vector_store.pkl"

def load_or_create_vectorstore():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    # Create a dummy document to initialize the vector store
    dummy_doc = Document(page_content="dummy")
    return FAISS.from_documents([dummy_doc], embeddings)

def update_vectorstore(faiss_index, new_text):
    documents = splitter.split_documents([Document(page_content=new_text)])
    faiss_index.add_documents(documents)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(faiss_index, f)
    return faiss_index

def get_relevant_chunks(faiss_index, query):
    return faiss_index.similarity_search(query, k=3)

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

def load_documents(docs_dir: str):
    documents = []
    docs_path = Path(docs_dir)

    for file_path in docs_path.glob("*"):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding = "utf-8")
            documents.extend(loader.load())

        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    return documents

def main():
    print("Loading documents...")
    documents = load_documents(DOCS_DIR)

    if not documents:
        print("No documents found. Add .txt or .pdf file inside data/docs/")
        return
    print(f"Loaded {len(documents)} document pages/files")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 120
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = CHROMA_DIR,
        collection_name = "naive_rag_collection"
    )
    
    print("Vector database created successfully")
    print(f"Saved to: {CHROMA_DIR}")

if __name__=="__main__":
    main()
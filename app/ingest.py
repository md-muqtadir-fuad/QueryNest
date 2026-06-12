import os
import time
import shutil
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

#Gemini free tier limit
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "80"))
EMBED_SLEEP_SECONDS = int(os.getenv("EMBED_SLEEP_SECONDS", "65"))

#First time rebuild cleanup
RESET_DB = os.getenv("RESET_DB", "true").lower() == "true"

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
        chunk_size=1200,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    if RESET_DB and Path(CHROMA_DIR).exists():
        print("Removing old Chroma database...")
        shutil.rmtree(CHROMA_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL
    )

    vector_store = Chroma(
        embedding_function = embeddings,
        persist_directory = CHROMA_DIR,
        collection_name = "querynest_collection"
    )

    total_chunks = len(chunks)

    for start in range(0, total_chunks, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        print(f"Embedding chunks start from {start+1} to {end} of {total_chunks} chunks...")

        ids = [f"chunk_{i}" for i in range (start, end)]

        vector_store.add_documents(
            documents = batch,
            ids = ids
        )

        if end < total_chunks:
            print(f"Speeping {EMBED_SLEEP_SECONDS} seconds to avaoid rate limit")
            time.sleep(EMBED_SLEEP_SECONDS)
    print("Vector database created successfully")
    print(f"Saved to: {CHROMA_DIR}")

if __name__=="__main__":
    main()
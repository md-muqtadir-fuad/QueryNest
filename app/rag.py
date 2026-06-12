'''
Naive RAG system for document-based QnA
'''

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR","./chroma_db")
GENERATION_MODEL = os.getenv("GENERATION_MODEL","gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","models/gemini-embedding-001")

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model = EMBEDDING_MODEL
    )
    vector_store = Chroma(
        persist_directory = CHROMA_DIR,
        embedding_function = embeddings,
        collection_name = "querynest_collection"
    )

    return vector_store

def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknwon")
        page = doc.metadata.get("page", "N/A")

        formatted.append(
            f"[Source {i}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )


    return "\n\n".join(formatted)

def extract_text_from_response(response):
    content = response.content

    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        texts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))

        return "\n".join(texts).strip()
    
    return str(content)

def ask_rag(question: str):
    vector_store = get_vector_store()

    docs = vector_store.similarity_search(
        question,
        k = 3
    )

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
         """
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.

Rules:
- If the answer is not in the context, say: "I do not know from the given documents."
- Do not invent information.
- Keep the answer clear and simple.
- Mention the source file if possible.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGoogleGenerativeAI(
        model = GENERATION_MODEL,
        temperature = 0.2,
        max_tokens = 500,
        max_retries = 2
    )

    message = prompt.format_messages(
        context = context,
        question = question
    )

    response = llm.invoke(
        message
    )

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:200]
        })

    return {
        "question": question,
        "answer": extract_text_from_response(response),
        "sources": sources
    }

if __name__ == "__main__":
    result = ask_rag("When was the university establish?")
    print(result["answer"])
    print(result["sources"])
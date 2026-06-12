from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask_rag

app = FastAPI(
    title = ""
)
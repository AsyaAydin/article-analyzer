import os
from contextlib import asynccontextmanager

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastembed import TextEmbedding
from openai import OpenAI
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from article_analyzer.answer import LLM_BASE_URL, answer
from article_analyzer.search import MODEL_NAME, search

load_dotenv()
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the model a single time, not per request
    state["model"] = TextEmbedding(MODEL_NAME)
    state["client"] = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
    yield


app = FastAPI(title="article-analyzer", lifespan=lifespan)


class Question(BaseModel):
    question: str
    k: int = 5


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


@app.get("/health")
def health():
    with connect() as conn:
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": n}


@app.post("/search")
def search_endpoint(q: Question):
    with connect() as conn:
        hits = search(conn, state["model"], q.question, q.k)
    return {"hits": hits}


@app.post("/ask")
def ask(q: Question):
    with connect() as conn:
        hits = search(conn, state["model"], q.question, q.k)
    result = answer(state["client"], q.question, hits)
    return {
        "answer": result["text"],
        "sources": [
            {
                "n": i,
                "arxiv_id": h["arxiv_id"],
                "title": h["title"],
                "section": h["section"],
                "url": f"https://arxiv.org/abs/{h['arxiv_id']}",
            }
            for i, h in enumerate(hits, start=1)
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "seconds": round(result["seconds"], 2),
        },
    }
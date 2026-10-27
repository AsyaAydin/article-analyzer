import os
import time

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from pgvector.psycopg import register_vector

from article_analyzer.search import MODEL_NAME, search
load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a research assistant for 21 cm cosmology.
Answer the question using ONLY the text of the numbered sources provided.
Paper titles and section names are context, NOT evidence: never state something
that appears only in a title. Only state what the source text explicitly says.
After every claim, cite the source(s) it comes from, like [1] or [2][4].
If the sources only partially answer the question, answer that part and say
clearly what the sources do not cover. Do not use outside knowledge."""


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {h['title']} (arXiv:{h['arxiv_id']}), section: {h['section']}\n"
            f"{h['content']}"
        )
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(client: OpenAI, question: str, hits: list[dict]) -> dict:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
    )
    return {
        "text": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    import sys

    load_dotenv()
    question = sys.argv[1]
    model = TextEmbedding(MODEL_NAME)
    client = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        hits = search(conn, model, question)

    result = answer(client, question, hits)

    print(result["text"])
    print("\nSources:")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h['title']} — https://arxiv.org/abs/{h['arxiv_id']} ({h['section']})")
    print(
        f"\n{result['prompt_tokens']} prompt + {result['completion_tokens']} completion tokens, "
        f"{result['seconds']:.1f}s"
    )
import os

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SEARCH = """
    SELECT c.arxiv_id, p.title, c.section, c.content,
           c.embedding <=> %(q)s AS distance
    FROM chunks c
    JOIN papers p ON p.arxiv_id = c.arxiv_id
    ORDER BY c.embedding <=> %(q)s
    LIMIT %(k)s
"""


def search(
    conn, model: TextEmbedding, question: str, k: int = 5, per_paper: int = 2
) -> list[dict]:
    query_vector = list(model.embed([QUERY_PREFIX + question]))[0]
    rows = conn.execute(SEARCH, {"q": query_vector, "k": k * 6}).fetchall()

    hits, counts = [], {}
    for a, t, s, c, d in rows:
        if counts.get(a, 0) >= per_paper:
            continue
        counts[a] = counts.get(a, 0) + 1
        hits.append({"arxiv_id": a, "title": t, "section": s, "content": c, "distance": d})
        if len(hits) == k:
            break
    return hits

if __name__ == "__main__":
    import sys

    load_dotenv()
    question = sys.argv[1]
    model = TextEmbedding(MODEL_NAME)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        for i, hit in enumerate(search(conn, model, question), start=1):
            print(f"\n#{i}  distance {hit['distance']:.3f}  |  {hit['arxiv_id']}")
            print(f"    {hit['title']}")
            print(f"    [{hit['section']}]")
            print(f"    {hit['content'][:300]}...")
import os
import time

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 256

SELECT_BATCH = """
    SELECT c.id, p.title, c.section, c.content
    FROM chunks c
    JOIN papers p ON p.arxiv_id = c.arxiv_id
    WHERE c.embedding IS NULL
    ORDER BY c.id
    LIMIT %s
"""
UPDATE_EMBEDDING = "UPDATE chunks SET embedding = %s WHERE id = %s"
COUNT_MISSING = "SELECT count(*) FROM chunks WHERE embedding IS NULL"


def chunk_text(title: str, section: str, content: str) -> str:
    """The text we actually embed: context header + chunk body."""
    return f"[{title} | {section}] {content}"


def embed_missing(conn, model: TextEmbedding, limit: int | None = None) -> None:
    missing = conn.execute(COUNT_MISSING).fetchone()[0]
    todo = missing if limit is None else min(limit, missing)
    print(f"{missing} chunks without embedding, embedding {todo}")

    done = 0
    start = time.perf_counter()
    while done < todo:
        rows = conn.execute(SELECT_BATCH, (min(BATCH_SIZE, todo - done),)).fetchall()
        if not rows:
            break

        texts = [chunk_text(title, section, content) for _, title, section, content in rows]
        vectors = list(model.embed(texts, batch_size=8))

        with conn.cursor() as cur:
            cur.executemany(
                UPDATE_EMBEDDING,
                [(vector, row[0]) for vector, row in zip(vectors, rows)],
            )
        conn.commit()

        done += len(rows)
        rate = done / (time.perf_counter() - start)
        eta_min = (todo - done) / rate / 60
        print(f"[{done}/{todo}] {rate:.1f} chunks/s, ~{eta_min:.0f} min left")


if __name__ == "__main__":
    import sys

    load_dotenv()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    model = TextEmbedding(MODEL_NAME)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        embed_missing(conn, model, limit)
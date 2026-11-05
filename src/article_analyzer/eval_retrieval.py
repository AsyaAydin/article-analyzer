import json
import os
from pathlib import Path
import sys
import time

from openai import OpenAI

from article_analyzer.answer import LLM_BASE_URL
from article_analyzer.rewrite import rewrite
import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

from article_analyzer.search import MODEL_NAME, QUERY_PREFIX

TOP_CHUNKS = 200
RANK_CHUNKS = "SELECT arxiv_id FROM chunks ORDER BY embedding <=> %s LIMIT %s"


def load_questions(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_ids(conn, questions: list[dict]) -> None:
    known = {row[0] for row in conn.execute("SELECT arxiv_id FROM papers")}
    for q in questions:
        for a in q["relevant"]:
            if a not in known:
                raise ValueError(f"{q['id']}: paper {a} is not in the corpus (typo?)")


def paper_ranking(conn, vector) -> list[str]:
    """Papers ordered by their best chunk."""
    ranking = []
    for (arxiv_id,) in conn.execute(RANK_CHUNKS, (vector, TOP_CHUNKS)):
        if arxiv_id not in ranking:
            ranking.append(arxiv_id)
    return ranking


def evaluate(conn, model, questions: list[dict], client: OpenAI | None = None) -> list[dict]:
    results = []
    for q in questions:
        text = q["question"]
        if client is not None:
            text = rewrite(client, text)
            print(f"{q['id']}  -> {text}")
            time.sleep(2)
        vector = list(model.embed([QUERY_PREFIX + text]))[0]
        ranking = paper_ranking(conn, vector)
        ranks = sorted(ranking.index(a) + 1 for a in q["relevant"] if a in ranking)
        first = ranks[0] if ranks else None
        results.append(
            {
                "id": q["id"],
                "first_rank": first,
                "hit@5": first is not None and first <= 5,
                "recall@10": sum(r <= 10 for r in ranks) / len(q["relevant"]),
                "rr": 1 / first if first else 0.0,
            }
        )
    return results


if __name__ == "__main__":
    load_dotenv()
    questions = load_questions(Path("eval/questions.jsonl"))
    model = TextEmbedding(MODEL_NAME)
    use_rewrite = "--rewrite" in sys.argv
    client = (
        OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"])
        if use_rewrite
        else None
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        conn.execute("SET hnsw.ef_search = 200")
        check_ids(conn, questions)
        results = evaluate(conn, model, questions, client)
    for r in results:
        rank = r["first_rank"] or "-"
        print(f"{r['id']}  first rank {rank:>3}  hit@5 {'yes' if r['hit@5'] else 'no ':>3}  recall@10 {r['recall@10']:.2f}")

    groups = {"expert (q)": "q", "plain (p)": "p"}
    for name, prefix in groups.items():
        part = [r for r in results if r["id"].startswith(prefix)]
        if not part:
            continue
        n = len(part)
        print(
            f"\n{name}: {n} questions | "
            f"hit@5 {sum(r['hit@5'] for r in part) / n:.2f} | "
            f"recall@10 {sum(r['recall@10'] for r in part) / n:.2f} | "
            f"MRR {sum(r['rr'] for r in part) / n:.2f}"
        )
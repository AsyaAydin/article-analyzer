import json
import os

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from pgvector.psycopg import register_vector

from article_analyzer.answer import LLM_BASE_URL, LLM_MODEL
from article_analyzer.search import MODEL_NAME, search

load_dotenv()
MAX_STEPS = 6

SYSTEM_PROMPT = """You are a research assistant for a corpus of 21 cm cosmology papers.
Use the tools to find evidence before answering. You may call several tools in a row.
Answer only from tool results, and cite each claim with the paper's arXiv id, e.g. [2408.14588].
If the tools return nothing useful, say so instead of guessing."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_passages",
            "description": "Semantic search over the papers' text. Use for questions about methods, results or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, in the field's terminology"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_papers",
            "description": "Find papers by author name, a keyword in the title, and/or publication year range. Returns title, authors, date and the start of the abstract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "author": {"type": "string"},
                    "title_keyword": {"type": "string"},
                    "year_from": {"type": "integer"},
                    "year_to": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper",
            "description": "Get title, authors, date and abstract of one paper by its arXiv id.",
            "parameters": {
                "type": "object",
                "properties": {"arxiv_id": {"type": "string"}},
                "required": ["arxiv_id"],
            },
        },
    },
]


def search_passages(conn, model, query: str) -> list[dict]:
    hits = search(conn, model, query, k=5)
    return [
        {"arxiv_id": h["arxiv_id"], "title": h["title"], "section": h["section"], "text": h["content"]}
        for h in hits
    ]


def find_papers(conn, author=None, title_keyword=None, year_from=None, year_to=None) -> list[dict]:
    conditions, params = [], []
    if author:
        conditions.append("EXISTS (SELECT 1 FROM unnest(authors) a WHERE a ILIKE %s)")
        params.append(f"%{author}%")
    if title_keyword:
        conditions.append("title ILIKE %s")
        params.append(f"%{title_keyword}%")
    if year_from:
        conditions.append("published >= make_date(%s, 1, 1)")
        params.append(year_from)
    if year_to:
        conditions.append("published <= make_date(%s, 12, 31)")
        params.append(year_to)
    where = " AND ".join(conditions) or "TRUE"
    rows = conn.execute(
        f"SELECT arxiv_id, title, authors, published, abstract FROM papers WHERE {where} "
        "ORDER BY published DESC LIMIT 20",
        params,
    ).fetchall()
    return [
        {"arxiv_id": a, "title": t, "authors": au[:5], "published": str(p), "abstract": ab[:600]}
        for a, t, au, p, ab in rows
    ]

def get_paper(conn, arxiv_id: str) -> dict:
    row = conn.execute(
        "SELECT arxiv_id, title, authors, published, abstract FROM papers WHERE arxiv_id = %s",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return {"error": f"paper {arxiv_id} is not in the corpus"}
    a, t, au, p, ab = row
    return {"arxiv_id": a, "title": t, "authors": au, "published": str(p), "abstract": ab}


def run_tool(conn, model, name: str, arguments: str):
    try:
        args = json.loads(arguments or "{}")
        if name == "search_passages":
            return search_passages(conn, model, **args)
        if name == "find_papers":
            return find_papers(conn, **args)
        if name == "get_paper":
            return get_paper(conn, **args)
        return {"error": f"unknown tool {name}"}
    except (json.JSONDecodeError, TypeError) as error:
        return {"error": f"bad arguments for {name}: {error}"}


def run_agent(conn, model, client: OpenAI, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=TOOLS, temperature=0
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in msg.tool_calls
                ],
            }
        )
        for call in msg.tool_calls:
            print(f"  step {step}: {call.function.name}({call.function.arguments})")
            result = run_tool(conn, model, call.function.name, call.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
            )
    messages.append(
        {"role": "user", "content": "Step limit reached. Answer now using only the information you already have."}
    )
    response = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, tools=TOOLS, tool_choice="none", temperature=0
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    import sys

    question = sys.argv[1]
    model = TextEmbedding(MODEL_NAME)
    client = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ["GROQ_API_KEY"])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        register_vector(conn)
        print(run_agent(conn, model, client, question))
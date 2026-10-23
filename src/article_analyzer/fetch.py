import os
import re

import arxiv
import psycopg
from dotenv import load_dotenv

# ---------- Search ----------

ID_PATTERN = re.compile(r"^(?P<base>.+?)v(?P<version>\d+)$")


def split_arxiv_id(short_id: str) -> tuple[str, int]:
    """'2301.01234v2' -> ('2301.01234', 2)"""
    match = ID_PATTERN.match(short_id)
    if match is None:
        raise ValueError(f"Unexpected arXiv id format: {short_id}")
    return match["base"], int(match["version"])


def search_papers(query: str, max_results: int) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for result in client.results(search):
        arxiv_id, version = split_arxiv_id(result.get_short_id())
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.date(),
                "pdf_url": result.pdf_url,
            }
        )
    return papers


# ---------- Save ----------

INSERT_PAPER = """
    INSERT INTO papers
        (arxiv_id, version, title, authors, abstract, categories, published, pdf_url)
    VALUES
        (%(arxiv_id)s, %(version)s, %(title)s, %(authors)s,
         %(abstract)s, %(categories)s, %(published)s, %(pdf_url)s)
    ON CONFLICT (arxiv_id) DO NOTHING
"""


def save_papers(conn, papers: list[dict]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for paper in papers:
            cur.execute(INSERT_PAPER, paper)
            inserted += cur.rowcount
    conn.commit()
    return inserted


# ---------- Run ----------

if __name__ == "__main__":
    load_dotenv()
    query = 'cat:astro-ph.CO AND abs:"21 cm"'

    papers = search_papers(query, max_results=50)
    print(f"Found {len(papers)} papers")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        new = save_papers(conn, papers)
        print(f"Inserted {new} new papers")


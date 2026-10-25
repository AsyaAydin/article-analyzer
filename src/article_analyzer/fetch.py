import os
import re
import time
from pathlib import Path

import httpx
import arxiv
import psycopg
from dotenv import load_dotenv

QUERY = (
    '(cat:astro-ph.CO OR cat:astro-ph.IM) AND abs:"21 cm" AND ('
    'submittedDate:[202301010000 TO 202612312359] OR '
    'abs:"simulation-based inference" OR abs:"neural" OR abs:"machine learning" OR '
    'abs:"likelihood-free" OR abs:"emulator" OR abs:"deep learning" OR '
    'abs:"normalizing flow" OR abs:"Bayesian")'
)
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

def count_matches(query: str, limit: int = 2000) -> int:
    """How many papers match the query (capped at `limit`), metadata only."""
    client = arxiv.Client(page_size=500)
    search = arxiv.Search(query=query, max_results=limit)
    return sum(1 for _ in client.results(search))
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


# ---------- Download ----------

SELECT_MISSING = "SELECT arxiv_id, pdf_url FROM papers WHERE pdf_path IS NULL ORDER BY arxiv_id"
UPDATE_PATH = "UPDATE papers SET pdf_path = %s WHERE arxiv_id = %s"


def download_missing_pdfs(conn, pdf_dir: Path, limit: int | None = None) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(SELECT_MISSING).fetchall()
    if limit is not None:
        rows = rows[:limit]
    print(f"{len(rows)} PDFs to download")

    for i, (arxiv_id, pdf_url) in enumerate(rows, start=1):
        path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        if path.exists():
            print(f"[{i}/{len(rows)}] {arxiv_id} already on disk")
        else:
            tmp_path = path.with_suffix(".pdf.part")
            try:
                response = httpx.get(pdf_url, follow_redirects=True, timeout=60)
                response.raise_for_status()
            except httpx.HTTPError as error:
                print(f"[{i}/{len(rows)}] {arxiv_id} FAILED: {error}")
                time.sleep(3)
                continue
            if not response.content.startswith(b"%PDF"):
                print(f"[{i}/{len(rows)}] {arxiv_id} SKIPPED: not a PDF")
                continue
            
            tmp_path.write_bytes(response.content)
            tmp_path.rename(path)
            print(f"[{i}/{len(rows)}] {arxiv_id} downloaded ({len(response.content) // 1024} KB)")
            time.sleep(3)

        conn.execute(UPDATE_PATH, (str(path), arxiv_id))
        conn.commit()

# ---------- Run ----------

if __name__ == "__main__":
    load_dotenv()
    query = 'cat:astro-ph.CO AND abs:"21 cm"'

    papers = search_papers(QUERY, max_results=1000)
    print(f"Found {len(papers)} papers")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        new = save_papers(conn, papers)
        print(f"Inserted {new} new papers")
        download_missing_pdfs(conn, Path("data/pdfs"))


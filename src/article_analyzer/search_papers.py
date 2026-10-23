import re
import time
from pathlib import Path

import arxiv

# Matches '2301.01234v2' or old-style 'astro-ph/0601001v1'
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

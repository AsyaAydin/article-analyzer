def save_papers(conn, papers: list[dict]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for paper in papers:
            cur.execute(INSERT_PAPER, paper)
            inserted += cur.rowcount
    conn.commit()
    return inserted

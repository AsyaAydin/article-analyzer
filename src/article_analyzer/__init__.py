import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    url = os.environ["DATABASE_URL"]
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT %s::vector <=> %s::vector",
            ("[1,2,3]", "[1,2,4]"),
        ).fetchone()
        print(f"Cosine distance: {row[0]}")

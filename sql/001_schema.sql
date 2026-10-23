CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id     TEXT PRIMARY KEY,         -- base id, e.g. '2301.01234'
    version      INT NOT NULL,             -- e.g. 2 for v2
    title        TEXT NOT NULL,
    authors      TEXT[] NOT NULL,          -- Postgres array type
    abstract     TEXT NOT NULL,
    categories   TEXT[] NOT NULL,
    published    DATE NOT NULL,
    pdf_url      TEXT NOT NULL,
    pdf_path     TEXT,                     -- NULL until downloaded
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

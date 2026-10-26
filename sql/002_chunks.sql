CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    arxiv_id     TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,          -- 0, 1, 2, ... within the paper
    section      TEXT NOT NULL,         -- e.g. '3.2 Training the network'
    content      TEXT NOT NULL,         -- the chunk text itself
    n_tokens     INT NOT NULL,
    embedding    vector(384),           -- NULL until step 3
    UNIQUE (arxiv_id, chunk_index)
);
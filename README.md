# article-analyzer

Search and ask questions about 21-cm cosmology papers from arXiv.

I wanted a way to look up methods and results across papers without opening every PDF again. This project downloads papers, extracts and splits their text, stores embeddings in PostgreSQL, and uses the retrieved passages to answer questions. The answer endpoint also returns the retrieved paper references so the original sources are easy to check.

There is also a separate tool-calling agent for more involved searches. No web UI, the interface is a small FastAPI service and a few command-line scripts.

## How it works

```text
Indexing:  arXiv -> PDFs -> Markdown -> chunks -> embeddings -> pgvector

Querying: question -> FastAPI -> vector search -> LLM -> answer + sources
                                     |
                                     +-> raw results (/search)
```

The ingestion steps live in separate scripts. Papers are stored with their arXiv IDs, titles, authors, abstracts and publication dates. PDF text is cleaned before chunking, chunks retain their section names and are embedded using `BAAI/bge-small-en-v1.5` (384 dimensions). Retrieval uses cosine distance in pgvector, with an HNSW index.

For answers, the service sends the retrieved passages to an LLM through an OpenAI-compatible API (Groq by default). The prompt asks the model to stick to those passages and cite them. The API returns both the generated text and source metadata, with token counts and request time. These are prompt level grounding rules.

A separate CLI agent in `agent.py` can search passages, find papers by author/title/year, and retrieve paper metadata by arXiv ID. It is not part of the FastAPI endpoints.

## Stack

- **Python / FastAPI** — API and ingestion scripts
- **PostgreSQL + pgvector** — paper metadata, chunks and vector search
- **FastEmbed / BGE** — local text embeddings
- **OpenAI-compatible client** — LLM calls (Groq endpoint by default)
- **PyMuPDF / pymupdf4llm** — PDF processing
- **Docker Compose** — API and database for local use
- **uv, pytest, GitHub Actions** — dependencies, tests and CI

## Run locally

You'll need Python 3.14, [uv](https://docs.astral.sh/uv/), Docker with Compose, and an API key for the configured LLM endpoint.

```bash
git clone https://github.com/AsyaAydin/article-analyzer.git
cd article-analyzer
uv sync
```

Create a `.env` file in the repository root:

```dotenv
DATABASE_URL=postgresql://postgres:dev@localhost:5432/postgres
GROQ_API_KEY=your_api_key_here
# Optional:
# LLM_BASE_URL=https://api.groq.com/openai/v1
# LLM_MODEL=openai/gpt-oss-120b
```

Start PostgreSQL and create the tables and vector index:

```bash
docker compose up -d db

# Run these from a Bash compatible shell.
for file in sql/*.sql; do
    docker compose exec -T db psql -U postgres -d postgres < "$file"
done
```

The database starts empty. To populate it, run the ingestion scripts in order:

```bash
uv run python -m article_analyzer.fetch
uv run --with pymupdf4llm python -m article_analyzer.parse
uv run python -m article_analyzer.chunk
uv run python -m article_analyzer.embed
```

`fetch.py` currently requests up to 1000 matching arXiv records and downloads their PDFs. For a smaller first run, change its query and `max_results` setting before running it. PDF processing and embedding can take a while on a larger collection.

`pymupdf4llm` is used by the parser but is not yet declared in `pyproject.toml`, the `--with` flag above provides it for that command. It should be added as a regular dependency before relying on unattended builds or CI.

Start the API:

```bash
docker compose up -d --build api
curl http://localhost:8000/health
```

The API runs on `http://localhost:8000`. FastAPI's interactive API docs are at `http://localhost:8000/docs`.

## API

Search without generating an answer:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"21-cm power spectrum emulators","k":5}'
```

Ask a question using retrieved passages:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How is simulation-based inference used in 21-cm cosmology?","k":5}'
```

`/search` returns matching passages and their metadata. `/ask` returns the answer, a numbered list of paper sources, and token/time usage. `/health` checks the database connection and reports the number of stored chunks.

The CLI agent can also be run directly once the database is populated:

```bash
uv run python -m article_analyzer.agent "Find papers using neural methods for 21-cm inference"
```

## Tests and retrieval evaluation

Unit tests cover arXiv ID parsing, text cleanup and chunking. Run them with:

```bash
uv run --with pymupdf4llm pytest -q
```

The GitHub Actions workflow runs tests and attempts a Docker image build on pushes and pull requests. It does not deploy the application. The parser dependency mentioned above needs to be recorded in the project dependencies for the unmodified CI workflow to run those tests reliably.

There is also a small labeled question set in `eval/questions.jsonl`. The evaluation script reports **hit@5**, **recall@10**, and **mean reciprocal rank** for retrieved papers, with expert-phrased and plain-language questions reported separately:

```bash
uv run python -m article_analyzer.eval_retrieval
```

Add `--rewrite` to compare retrieval after LLM-based query rewriting. This evaluates retrieval against the labeled paper set, it is not an evaluation of factual accuracy in generated answers.

## Repository layout

```text
src/article_analyzer/
  fetch.py            arXiv metadata and PDF downloads
  parse.py            PDF to cleaned Markdown
  chunk.py            section-aware text chunks
  embed.py            embedding and database updates
  search.py           vector search
  answer.py           grounded answer generation
  agent.py            tool-calling CLI agent
  api.py              FastAPI endpoints
  rewrite.py          query rewriting
  eval_retrieval.py   retrieval evaluation
sql/                  database tables and HNSW index
tests/                unit tests
eval/                 labeled retrieval questions
.github/workflows/    CI workflow
Dockerfile            API image
compose.yaml          local API + PostgreSQL services
```


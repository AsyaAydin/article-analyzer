FROM python:3.14-slim

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "article_analyzer.api:app", "--host", "0.0.0.0", "--port", "8000"]
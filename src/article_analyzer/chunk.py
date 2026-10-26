import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from tokenizers import Tokenizer

TOKENIZER = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

MAX_TOKENS = 350
OVERLAP_TOKENS = 50
MIN_TOKENS = 25

HEADING = re.compile(r"^#{1,6}\s+(.*)$", flags=re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SKIP_SECTIONS = re.compile(
    r"acknowledg|funding|competing interest|data availability|author contribution",
    flags=re.IGNORECASE,
)


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)

def strip_tables(md: str) -> str:
    """Remove Markdown table rows (lines starting with '|'). Captions stay."""
    return "\n".join(
        line for line in md.split("\n") if not line.lstrip().startswith("|")
    )

def split_by_tokens(text: str, size: int) -> list[str]:
    """Cut text into pieces of at most `size` tokens, at exact token boundaries."""
    enc = TOKENIZER.encode(text, add_special_tokens=False)
    pieces = []
    for i in range(0, len(enc.ids), size):
        start = enc.offsets[i][0]
        end = enc.offsets[min(i + size, len(enc.ids)) - 1][1]
        pieces.append(text[start:end])
    return pieces

def split_sections(md: str) -> list[tuple[str, str]]:
    """Split Markdown at headings -> [(section_title, text), ...]."""
    matches = list(HEADING.finditer(md))
    sections = []

    first_start = matches[0].start() if matches else len(md)
    if md[:first_start].strip():
        sections.append(("Front matter", md[:first_start]))

    for match, nxt in zip(matches, matches[1:] + [None]):
        end = nxt.start() if nxt else len(md)
        title = match.group(1).replace("*", "").strip()
        sections.append((title, md[match.end() : end]))
    return sections


def split_units(text: str) -> list[tuple[str, int]]:
    """Split text into sentences with their token counts.
    Sentences longer than MAX_TOKENS are cut into word pieces."""
    units = []
    for sentence in SENTENCE_END.split(text):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        n = count_tokens(sentence)
        if n <= MAX_TOKENS:
            units.append((sentence, n))
        else:
            for piece in split_by_tokens(sentence, MAX_TOKENS):
                units.append((piece, count_tokens(piece)))
    return units


def pack(units: list[tuple[str, int]]) -> list[str]:
    """Greedily pack sentences into chunks of ~MAX_TOKENS,
    starting each new chunk with the last ~OVERLAP_TOKENS of the previous one."""
    chunks = []
    current, current_tokens = [], 0

    for sentence, n in units:
        if current and current_tokens + n > MAX_TOKENS:
            chunks.append(" ".join(s for s, _ in current))

            overlap, overlap_tokens = [], 0
            for prev, t in reversed(current):
                if overlap_tokens + t > OVERLAP_TOKENS:
                    break
                overlap.insert(0, (prev, t))
                overlap_tokens += t
            current, current_tokens = overlap, overlap_tokens

        current.append((sentence, n))
        current_tokens += n

    if current:
        chunks.append(" ".join(s for s, _ in current))
    return chunks


def chunk_paper(md: str) -> list[dict]:
    md = strip_tables(md)
    chunks = []
    for section, text in split_sections(md):
        if SKIP_SECTIONS.search(section):
            continue
        for body in pack(split_units(text)):
            n = count_tokens(body)
            if n >= MIN_TOKENS:
                chunks.append({"section": section, "content": body, "n_tokens": n})
    return chunks


DELETE_CHUNKS = "DELETE FROM chunks WHERE arxiv_id = %s"
INSERT_CHUNK = """
    INSERT INTO chunks (arxiv_id, chunk_index, section, content, n_tokens)
    VALUES (%s, %s, %s, %s, %s)
"""


def chunk_all(conn, md_dir: Path) -> None:
    paths = sorted(md_dir.glob("*.md"))
    total = 0
    for i, path in enumerate(paths, start=1):
        arxiv_id = path.stem.replace("_", "/")
        md = path.read_text(encoding="utf-8").replace("\x00", "")
        chunks = chunk_paper(md)
        rows = [
            (arxiv_id, j, c["section"], c["content"], c["n_tokens"])
            for j, c in enumerate(chunks)
        ]
        with conn.cursor() as cur:
            cur.execute(DELETE_CHUNKS, (arxiv_id,))
            cur.executemany(INSERT_CHUNK, rows)
        conn.commit()

        total += len(chunks)
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {total} chunks so far")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test mode: chunk one Markdown file and show the first chunks
        chunks = chunk_paper(Path(sys.argv[1]).read_text(encoding="utf-8").replace("\x00", ""))
        print(f"{len(chunks)} chunks")
        for c in chunks[:6]:
            print(f"\n--- [{c['section']}] {c['n_tokens']} tokens")
            print(c["content"][:300])
    else:
        load_dotenv()
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            chunk_all(conn, Path("data/markdown"))
import time
import traceback
from collections import Counter
from pathlib import Path
import re
import pymupdf4llm
import pymupdf


PICTURE_TEXT = re.compile(
    r"<!-- Start of picture text -->.*?<!-- End of picture text -->",
    flags=re.DOTALL,
)
REFERENCES_HEADING = re.compile(
    r"^#+\s*\**\s*(?:\d+\.?\s*)?(?:References|Bibliography)\b",
    flags=re.MULTILINE | re.IGNORECASE,
)

SLOW_PAPERS = {"2609.21479"}

def pdf_to_markdown(pdf_path: Path) -> str:
    return pymupdf4llm.to_markdown(str(pdf_path))

def pdf_to_text_fast(pdf_path: Path) -> str:
    """Fallback: plain text extraction, no layout model. Fast but rougher."""
    doc = pymupdf.open(pdf_path)
    return "\n\n".join(page.get_text() for page in doc)

def clean_markdown(md: str) -> str:
    # 1. Drop text extracted from inside figures
    md = PICTURE_TEXT.sub("", md)

    # 2. Cut everything from the References heading onwards
    match = REFERENCES_HEADING.search(md)
    if match:
        md = md[: match.start()]

    # 3. Turn HTML super/subscripts into plain notation: χ<sup>2</sup> -> χ^2
    md = re.sub(r"<sup>(.*?)</sup>", r"^\1", md)
    md = re.sub(r"<sub>(.*?)</sub>", r"_\1", md)

    # 4. Drop running headers/footers: short lines repeated on several pages,
    #    and lines that are only a page number
    lines = md.split("\n")
    counts = Counter(line.strip() for line in lines if line.strip())
    kept = []
    for line in lines:
        s = line.strip()
        if s.isdigit():
            continue
        if counts[s] >= 3 and len(s) < 100:
            continue
        kept.append(line)
    md = "\n".join(kept)

    # 5. Collapse runs of blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()



def parse_all(pdf_dir: Path, md_dir: Path) -> None:
    md_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    todo = [p for p in pdfs if not (md_dir / f"{p.stem}.md").exists()]
    print(f"{len(pdfs)} PDFs, {len(pdfs) - len(todo)} already parsed, {len(todo)} to do")

    failed = []
    for i, pdf in enumerate(todo, start=1):
        out = md_dir / f"{pdf.stem}.md"
        start = time.perf_counter()
        try:
            if pdf.stem in SLOW_PAPERS:
                print(f"[{i}/{len(todo)}] {pdf.stem} using fast fallback")
                clean = clean_markdown(pdf_to_text_fast(pdf))
            else:
                clean = clean_markdown(pdf_to_markdown(pdf))
        except Exception:
            print(f"[{i}/{len(todo)}] {pdf.stem} FAILED")
            traceback.print_exc()
            failed.append(pdf.stem)
            continue

        tmp = out.with_suffix(".md.part")
        tmp.write_text(clean, encoding="utf-8")
        tmp.rename(out)
        seconds = time.perf_counter() - start
        print(f"[{i}/{len(todo)}] {pdf.stem} {len(clean) // 1000}k chars, {seconds:.0f}s")

    print(f"Done. {len(failed)} failed: {failed}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test mode: one paper, print stats
        path = Path(sys.argv[1])
        raw = pdf_to_markdown(path)
        clean = clean_markdown(raw)
        print(f"raw:   {len(raw):>7} chars")
        print(f"clean: {len(clean):>7} chars ({100 * len(clean) / len(raw):.0f}% kept)")
        for word in ["Frequency (MHz)", "Publications of the Astronomical", "References"]:
            print(f"{word!r}: raw {raw.count(word)}, clean {clean.count(word)}")
        print("-" * 60)
        print(clean[-1500:])
    else:
        # Batch mode: all papers
        parse_all(Path("data/pdfs"), Path("data/markdown"))
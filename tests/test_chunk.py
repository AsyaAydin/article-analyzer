from article_analyzer.chunk import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    SENTENCE_END,
    chunk_paper,
    pack,
    split_units,
    strip_tables,
)


def test_table_rows_removed_caption_kept():
    md = "Table 1. Measured limits.\n|a|b|\n|1|2|\nNormal text after the table."
    out = strip_tables(md)
    assert "Table 1. Measured limits." in out
    assert "|a|b|" not in out
    assert "Normal text after the table." in out


def test_no_chunk_exceeds_limit_even_without_spaces():
    # A long run without spaces, like the tables that once produced 6650-token chunks
    no_spaces = "|".join(str(i) for i in range(3000))
    md = "## Results\n" + "This is a sentence. " * 200 + no_spaces
    for chunk in chunk_paper(md):
        assert chunk["n_tokens"] <= MAX_TOKENS + OVERLAP_TOKENS


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence number {i} is about reionization." for i in range(80))
    chunks = pack(split_units(text))
    assert len(chunks) >= 2
    last_sentence_of_first = SENTENCE_END.split(chunks[0])[-1]
    assert last_sentence_of_first in chunks[1]
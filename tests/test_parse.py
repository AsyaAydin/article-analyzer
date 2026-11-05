from article_analyzer.parse import clean_markdown

RAW = """## 1. Introduction
The 21 cm signal is faint.
<!-- Start of picture text -->
170 175 180 Frequency (MHz)
<!-- End of picture text -->
The χ<sup>2</sup> statistic is useful.
## References
[1] Someone, A. A paper title."""


def test_figure_text_removed():
    assert "Frequency (MHz)" not in clean_markdown(RAW)


def test_references_cut():
    assert "Someone" not in clean_markdown(RAW)


def test_superscript_converted():
    assert "χ^2" in clean_markdown(RAW)


def test_real_text_kept():
    assert "The 21 cm signal is faint." in clean_markdown(RAW)
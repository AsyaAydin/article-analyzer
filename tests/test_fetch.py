import pytest

from article_analyzer.fetch import split_arxiv_id


def test_new_style_id():
    assert split_arxiv_id("2301.01234v2") == ("2301.01234", 2)


def test_old_style_id():
    assert split_arxiv_id("astro-ph/0601001v1") == ("astro-ph/0601001", 1)


def test_id_without_version_is_rejected():
    with pytest.raises(ValueError):
        split_arxiv_id("2301.01234")
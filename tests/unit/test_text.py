import pytest

from paper_similarity.domain.text import build_embedding_text, content_hash, normalize_whitespace


def test_normalize_whitespace_collapses_line_breaks_and_runs() -> None:
    assert normalize_whitespace("  a\n  b\t\tc  ") == "a b c"


@pytest.mark.parametrize(
    ("title", "body", "expected"),
    [
        ("Deep Retrieval", "We study  retrieval.", "Deep Retrieval. We study retrieval."),
        ("Is it?", "Yes.", "Is it? Yes."),
        ("Ends with dot.", "Body", "Ends with dot. Body"),
        (None, "Only body", "Only body"),
        ("   ", "Only body", "Only body"),
    ],
)
def test_build_embedding_text(title: str | None, body: str, expected: str) -> None:
    assert build_embedding_text(title, body) == expected


def test_content_hash_is_stable_and_sensitive() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    assert len(content_hash("abc")) == 64

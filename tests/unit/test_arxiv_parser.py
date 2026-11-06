from datetime import UTC, datetime

import pytest

from paper_similarity.ingestion.parser import MalformedEntryError, parse_feed, split_arxiv_id
from tests.conftest import FixtureLoader


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2611.08433v1", ("2611.08433", 1)),
        ("https://arxiv.org/abs/2401.12345v12", ("2401.12345", 12)),
        ("http://arxiv.org/abs/2401.12345", ("2401.12345", 1)),
        ("http://arxiv.org/abs/hep-th/9901001v3", ("hep-th/9901001", 3)),
    ],
)
def test_split_arxiv_id_strips_version(raw: str, expected: tuple[str, int]) -> None:
    assert split_arxiv_id(raw) == expected


def test_split_arxiv_id_rejects_unknown_format() -> None:
    with pytest.raises(MalformedEntryError):
        split_arxiv_id("urn:something-else")


def test_parse_recorded_page(fixture_bytes: FixtureLoader) -> None:
    feed = parse_feed(fixture_bytes("arxiv_page.xml"))

    assert feed.error_message is None
    assert feed.skipped == []
    assert feed.total_results == 13294
    assert len(feed.papers) == 3

    paper = feed.papers[0]
    assert paper.arxiv_id == "2611.08433"
    assert paper.version == 1
    assert paper.title.startswith("MedGraph: Maintaining Patient State")
    assert paper.abstract.startswith("LLM-based medical assistants")
    assert paper.authors[0] == "Donghua Zhang"
    assert paper.categories == ("cs.CL", "cs.AI")
    assert paper.primary_category == "cs.CL"
    assert paper.published == datetime(2026, 11, 5, 18, 27, 20, tzinfo=UTC)
    assert paper.abs_url == "https://arxiv.org/abs/2611.08433"
    assert paper.pdf_url == "https://arxiv.org/pdf/2611.08433v1"


def test_malformed_entries_are_skipped_not_fatal(fixture_bytes: FixtureLoader) -> None:
    feed = parse_feed(fixture_bytes("arxiv_malformed.xml"))

    assert [p.arxiv_id for p in feed.papers] == ["hep-th/9901001"]
    assert len(feed.skipped) == 2
    assert any("missing abstract" in reason for reason in feed.skipped)
    assert any("unrecognised arXiv id" in reason for reason in feed.skipped)


def test_whitespace_is_normalised(fixture_bytes: FixtureLoader) -> None:
    paper = parse_feed(fixture_bytes("arxiv_malformed.xml")).papers[0]

    assert paper.title == "An Old-Style Identifier Paper"
    assert paper.abstract == "Abstract text spread over several lines."
    assert paper.version == 3
    assert paper.primary_category == "hep-th"  # falls back to the first category


def test_in_band_error_entry_is_detected(fixture_bytes: FixtureLoader) -> None:
    feed = parse_feed(fixture_bytes("arxiv_error_invalid_query.xml"))

    assert feed.papers == []
    assert feed.error_message is not None
    assert "Invalid query string" in feed.error_message


def test_empty_result_page(fixture_bytes: FixtureLoader) -> None:
    feed = parse_feed(fixture_bytes("arxiv_empty.xml"))

    assert feed.papers == []
    assert feed.total_results == 0
    assert feed.error_message is None

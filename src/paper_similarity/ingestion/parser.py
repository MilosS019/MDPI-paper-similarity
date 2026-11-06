"""Parse arXiv API responses (Atom XML) into domain objects.

The parser is pure (bytes in, objects out) so it can be tested against recorded responses.
One malformed entry is reported and skipped; it never fails the whole page.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import feedparser

from paper_similarity.domain.models import ArxivPaper
from paper_similarity.domain.text import normalize_whitespace

ARXIV_ERROR_ID = "arxiv.org/api/errors"

# New-style ids (2401.12345v2) and old-style ids (hep-th/9901001v1).
_ID_PATTERN = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<version>\d+))?$")


@dataclass(frozen=True, slots=True)
class ParsedFeed:
    papers: list[ArxivPaper]
    total_results: int
    # Human-readable reasons for entries that could not be parsed.
    skipped: list[str] = field(default_factory=list)
    # Set when arXiv answered with its in-band error entry instead of results.
    error_message: str | None = None


class MalformedEntryError(ValueError):
    pass


def split_arxiv_id(raw_id: str) -> tuple[str, int]:
    """``http://arxiv.org/abs/2401.12345v2`` -> ``("2401.12345", 2)``.

    The version is dropped from the identifier so that every version of a paper maps to
    the same record. A missing version is treated as v1.
    """
    match = _ID_PATTERN.search(raw_id.strip())
    if match is None:
        raise MalformedEntryError(f"unrecognised arXiv id {raw_id!r}")
    version = int(match.group("version")) if match.group("version") else 1
    return match.group("id"), version


def _parse_timestamp(value: str | None, field_name: str) -> datetime:
    if not value:
        raise MalformedEntryError(f"missing {field_name}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MalformedEntryError(f"invalid {field_name} {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_entry(entry: Any) -> ArxivPaper:
    arxiv_id, version = split_arxiv_id(entry.get("id", ""))
    title = normalize_whitespace(entry.get("title", ""))
    abstract = normalize_whitespace(entry.get("summary", ""))
    if not title:
        raise MalformedEntryError(f"{arxiv_id}: missing title")
    if not abstract:
        raise MalformedEntryError(f"{arxiv_id}: missing abstract")

    authors = tuple(
        name for a in entry.get("authors", []) if (name := normalize_whitespace(a.get("name", "")))
    )
    categories = tuple(t["term"] for t in entry.get("tags", []) if t.get("term"))
    primary = entry.get("arxiv_primary_category", {}).get("term") or (
        categories[0] if categories else None
    )
    pdf_url = next(
        (
            link["href"]
            for link in entry.get("links", [])
            if link.get("type") == "application/pdf" or link.get("title") == "pdf"
        ),
        None,
    )
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=primary,
        published=_parse_timestamp(entry.get("published"), "published"),
        updated=_parse_timestamp(entry.get("updated") or entry.get("published"), "updated"),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url,
    )


def parse_feed(content: bytes) -> ParsedFeed:
    feed = feedparser.parse(content)
    if feed.get("bozo") and not feed.get("entries") and not feed.get("feed"):
        raise MalformedEntryError(
            f"response is not a valid Atom feed: {feed.get('bozo_exception')}"
        )

    entries = feed.get("entries", [])
    # arXiv reports some errors in-band: a single entry whose id points to /api/errors.
    if len(entries) == 1 and ARXIV_ERROR_ID in entries[0].get("id", ""):
        return ParsedFeed(
            papers=[],
            total_results=0,
            error_message=normalize_whitespace(entries[0].get("summary", "unknown error")),
        )

    total = int(feed.get("feed", {}).get("opensearch_totalresults", len(entries)) or 0)
    papers: list[ArxivPaper] = []
    skipped: list[str] = []
    for entry in entries:
        try:
            papers.append(_parse_entry(entry))
        except MalformedEntryError as exc:
            skipped.append(str(exc))
    return ParsedFeed(papers=papers, total_results=total, skipped=skipped)

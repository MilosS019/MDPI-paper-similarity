from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from paper_similarity.domain.models import ArxivPaper
from paper_similarity.ingestion.parquet_store import RawParquetStore
from paper_similarity.ingestion.raw_reader import RawParquetReader

BASE = ArxivPaper(
    arxiv_id="2401.00001",
    version=1,
    title="Original title",
    abstract="Original abstract.",
    authors=("A. Author",),
    categories=("cs.CL", "cs.IR"),
    primary_category="cs.CL",
    published=datetime(2024, 1, 1, tzinfo=UTC),
    updated=datetime(2024, 1, 1, tzinfo=UTC),
    abs_url="https://arxiv.org/abs/2401.00001",
    pdf_url=None,
)


def test_latest_version_of_each_paper_wins_across_runs(tmp_path: Path) -> None:
    store = RawParquetStore(tmp_path)
    v2 = replace(BASE, version=2, title="Revised title", updated=datetime(2024, 3, 1, tzinfo=UTC))
    other = replace(BASE, arxiv_id="2401.00002", primary_category="cs.IR")
    store.new_run("q", datetime(2026, 1, 1, tzinfo=UTC)).write_page([BASE, other])
    store.new_run("q", datetime(2026, 2, 1, tzinfo=UTC)).write_page([v2])

    papers = RawParquetReader(tmp_path).latest_papers()

    assert [(p.arxiv_id, p.version, p.title) for p in papers] == [
        ("2401.00001", 2, "Revised title"),
        ("2401.00002", 1, "Original title"),
    ]
    restored = papers[0]
    assert restored.authors == ("A. Author",)
    assert restored.categories == ("cs.CL", "cs.IR")
    assert restored.updated == datetime(2024, 3, 1, tzinfo=UTC)  # UTC, not local time
    assert restored.updated.utcoffset() is not None


def test_summary_counts_runs_and_categories(tmp_path: Path) -> None:
    store = RawParquetStore(tmp_path)
    store.new_run("q", datetime(2026, 1, 1, tzinfo=UTC)).write_page([BASE])
    store.new_run("q", datetime(2026, 2, 1, tzinfo=UTC)).write_page(
        [BASE, replace(BASE, arxiv_id="2401.00002", primary_category="cs.IR")]
    )

    summary = RawParquetReader(tmp_path).summary()

    assert summary is not None
    assert (summary.runs, summary.records, summary.papers) == (2, 3, 2)
    assert summary.top_categories == [("cs.CL", 1), ("cs.IR", 1)]


def test_empty_landing_zone(tmp_path: Path) -> None:
    reader = RawParquetReader(tmp_path / "missing")
    assert reader.latest_papers() == []
    assert reader.summary() is None

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from paper_similarity.domain.models import ArxivPaper
from paper_similarity.ingestion.parquet_store import RAW_SCHEMA, RawParquetStore

PAPER = ArxivPaper(
    arxiv_id="2401.00001",
    version=2,
    title="A title",
    abstract="An abstract.",
    authors=("A. Author", "B. Author"),
    categories=("cs.CL",),
    primary_category="cs.CL",
    published=datetime(2024, 1, 1, tzinfo=UTC),
    updated=datetime(2024, 2, 1, tzinfo=UTC),
    abs_url="https://arxiv.org/abs/2401.00001",
    pdf_url=None,
)


def test_pages_are_written_as_numbered_parts_in_a_run_partition(tmp_path: Path) -> None:
    writer = RawParquetStore(tmp_path).new_run(
        "cat:cs.CL", datetime(2026, 11, 6, 10, 0, tzinfo=UTC)
    )

    first = writer.write_page([PAPER])
    second = writer.write_page([PAPER])

    assert first == tmp_path / "run_ts=2026-11-06T10-00-00Z" / f"part-00000-{writer.run_id}.parquet"
    assert second is not None
    assert second.name == f"part-00001-{writer.run_id}.parquet"
    assert not list(tmp_path.rglob("*.tmp"))  # atomic rename left no temp files

    table = pq.read_table(first)
    assert table.schema.equals(RAW_SCHEMA)
    row = table.to_pylist()[0]
    assert row["arxiv_id"] == "2401.00001"
    assert row["authors"] == ["A. Author", "B. Author"]
    assert row["query"] == "cat:cs.CL"


def test_empty_pages_write_nothing(tmp_path: Path) -> None:
    writer = RawParquetStore(tmp_path).new_run("q")
    assert writer.write_page([]) is None
    assert not any(tmp_path.iterdir())


def test_runs_starting_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path) -> None:
    started = datetime(2026, 11, 6, 10, 0, tzinfo=UTC)
    store = RawParquetStore(tmp_path)
    first = store.new_run("query-a", started).write_page([PAPER])
    second = store.new_run("query-b", started).write_page([PAPER])

    assert first is not None
    assert second is not None
    assert first != second
    assert first.parent == second.parent
    assert {pq.read_table(f).column("query")[0].as_py() for f in (first, second)} == {
        "query-a",
        "query-b",
    }

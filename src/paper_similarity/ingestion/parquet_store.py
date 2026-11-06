"""Raw landing zone: every ingestion run writes what it fetched from arXiv to Parquet.

Layout (Hive-style partitions, one folder per run, one file per API page)::

    <root>/run_ts=2026-11-06T10-00-00Z/part-00000.parquet

Keeping the raw records means the database and vector index can be rebuilt (for example after
switching the embedding model) without calling arXiv again, and there is an audit trail of
what arXiv returned and when. Files are immutable: runs only ever add new files.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from paper_similarity.domain.models import ArxivPaper

RAW_SCHEMA = pa.schema(
    [
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("version", pa.int32(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("abstract", pa.string(), nullable=False),
        pa.field("authors", pa.list_(pa.string()), nullable=False),
        pa.field("categories", pa.list_(pa.string()), nullable=False),
        pa.field("primary_category", pa.string()),
        pa.field("published", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("updated", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("abs_url", pa.string(), nullable=False),
        pa.field("pdf_url", pa.string()),
        pa.field("query", pa.string(), nullable=False),
        pa.field("fetched_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def run_partition(started_at: datetime) -> str:
    return "run_ts=" + started_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


class RawRunWriter:
    """Writes the pages of one ingestion run into that run's partition folder."""

    def __init__(self, directory: Path, query: str) -> None:
        self.directory = directory
        self._query = query
        self._part = 0
        self.files: list[Path] = []

    def write_page(self, papers: Sequence[ArxivPaper]) -> Path | None:
        if not papers:
            return None
        fetched_at = datetime.now(UTC)
        table = pa.Table.from_pylist(
            [
                {
                    "arxiv_id": p.arxiv_id,
                    "version": p.version,
                    "title": p.title,
                    "abstract": p.abstract,
                    "authors": list(p.authors),
                    "categories": list(p.categories),
                    "primary_category": p.primary_category,
                    "published": p.published,
                    "updated": p.updated,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "query": self._query,
                    "fetched_at": fetched_at,
                }
                for p in papers
            ],
            schema=RAW_SCHEMA,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"part-{self._part:05d}.parquet"
        # Write to a temporary name and rename, so readers never see a half-written file.
        tmp = target.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(target)
        self._part += 1
        self.files.append(target)
        return target


class RawParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def new_run(self, query: str, started_at: datetime | None = None) -> RawRunWriter:
        partition = run_partition(started_at or datetime.now(UTC))
        return RawRunWriter(self.root / partition, query)

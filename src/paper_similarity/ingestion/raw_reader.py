"""Read the raw landing zone back with DuckDB.

DuckDB runs in-process and queries the Parquet files directly with SQL, without loading them
into a database first. The same queries work on ``s3://`` paths in production (httpfs).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from paper_similarity.domain.models import ArxivPaper

# Every run may have fetched the same paper; keep its most recent version.
# QUALIFY filters on a window function, like HAVING does for aggregates.
LATEST_PAPERS_SQL = """
SELECT arxiv_id, version, title, abstract, authors, categories, primary_category,
       published, updated, abs_url, pdf_url
FROM read_parquet($glob, hive_partitioning = true)
QUALIFY row_number() OVER (
    PARTITION BY arxiv_id ORDER BY version DESC, updated DESC, fetched_at DESC
) = 1
ORDER BY arxiv_id
"""

SUMMARY_SQL = """
SELECT count(DISTINCT run_ts)   AS runs,
       count(*)                 AS records,
       count(DISTINCT arxiv_id) AS papers,
       min(published)::DATE     AS oldest,
       max(published)::DATE     AS newest
FROM read_parquet($glob, hive_partitioning = true)
"""

CATEGORIES_SQL = """
SELECT primary_category, count(DISTINCT arxiv_id) AS papers
FROM read_parquet($glob, hive_partitioning = true)
GROUP BY primary_category
ORDER BY papers DESC, primary_category
LIMIT $top
"""


@dataclass(frozen=True, slots=True)
class RawSummary:
    runs: int
    records: int
    papers: int
    oldest: Any
    newest: Any
    top_categories: list[tuple[str | None, int]]


class RawParquetReader:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def _glob(self) -> str:
        return (self.root / "*" / "*.parquet").as_posix()

    def has_data(self) -> bool:
        return self.root.is_dir() and any(self.root.glob("*/*.parquet"))

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")  # otherwise timestamps come back in local time
        return con

    def latest_papers(self) -> list[ArxivPaper]:
        if not self.has_data():
            return []
        with self._connect() as con:
            rows = con.sql(LATEST_PAPERS_SQL, params={"glob": self._glob}).to_arrow_table()
        return [
            ArxivPaper(
                arxiv_id=r["arxiv_id"],
                version=r["version"],
                title=r["title"],
                abstract=r["abstract"],
                authors=tuple(r["authors"]),
                categories=tuple(r["categories"]),
                primary_category=r["primary_category"],
                published=r["published"],
                updated=r["updated"],
                abs_url=r["abs_url"],
                pdf_url=r["pdf_url"],
            )
            for r in rows.to_pylist()
        ]

    def summary(self, top: int = 10) -> RawSummary | None:
        if not self.has_data():
            return None
        with self._connect() as con:
            row = con.execute(SUMMARY_SQL, {"glob": self._glob}).fetchone()
            categories = con.execute(CATEGORIES_SQL, {"glob": self._glob, "top": top}).fetchall()
        runs, records, papers, oldest, newest = row or (0, 0, 0, None, None)
        return RawSummary(
            runs=int(runs),
            records=int(records),
            papers=int(papers),
            oldest=oldest,
            newest=newest,
            top_categories=[(c, int(n)) for c, n in categories],
        )

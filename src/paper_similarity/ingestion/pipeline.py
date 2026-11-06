"""The arXiv ingestion pipeline.

    arXiv API -> parse -> Parquet (raw) -> Postgres upsert -> embed -> Qdrant -> mark indexed

Each page is landed and committed on its own, so a failure halfway through keeps everything
that was already fetched. Indexing runs at the end of every run and also heals rows left
pending by earlier, interrupted runs.
"""

from dataclasses import dataclass, field
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paper_similarity.core import metrics
from paper_similarity.db.repositories import PaperRepository
from paper_similarity.domain.errors import ArxivUnavailableError
from paper_similarity.ingestion.arxiv_client import ArxivClient, SortBy
from paper_similarity.ingestion.parquet_store import RawParquetStore
from paper_similarity.services.indexing import IndexingService

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class IngestionReport:
    query: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    indexed: int = 0
    raw_files: list[Path] = field(default_factory=list)
    # Set when arXiv failed partway: everything fetched before the failure was still saved.
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class IngestionPipeline:
    def __init__(
        self,
        arxiv: ArxivClient,
        session_factory: async_sessionmaker[AsyncSession],
        raw_store: RawParquetStore,
        indexer: IndexingService,
    ) -> None:
        self._arxiv = arxiv
        self._session_factory = session_factory
        self._raw_store = raw_store
        self._indexer = indexer

    async def run(
        self, query: str, max_results: int, sort_by: SortBy = "submittedDate"
    ) -> IngestionReport:
        report = IngestionReport(query=query)
        writer = self._raw_store.new_run(query)
        log.info("ingestion_started", query=query, max_results=max_results)

        try:
            async for page in self._arxiv.search(query, max_results, sort_by=sort_by):
                report.fetched += len(page.papers)
                report.skipped += len(page.skipped)

                raw_file = writer.write_page(page.papers)
                if raw_file:
                    report.raw_files.append(raw_file)

                async with self._session_factory() as session, session.begin():
                    result = await PaperRepository(session).upsert(page.papers)
                report.inserted += result.inserted
                report.updated += result.updated
                report.unchanged += result.unchanged
                log.info(
                    "ingestion_page_stored",
                    start=page.start,
                    papers=len(page.papers),
                    inserted=result.inserted,
                    updated=result.updated,
                    unchanged=result.unchanged,
                )
        except ArxivUnavailableError as exc:
            # Keep and index what we already have; report the failure to the caller.
            report.error = str(exc)
            log.error("ingestion_arxiv_failed", error=report.error, fetched=report.fetched)

        report.indexed = await self._indexer.index_pending()

        for outcome in ("inserted", "updated", "unchanged", "skipped"):
            metrics.PAPERS_INGESTED.labels(outcome=outcome).inc(getattr(report, outcome))
        log.info(
            "ingestion_finished",
            fetched=report.fetched,
            inserted=report.inserted,
            updated=report.updated,
            unchanged=report.unchanged,
            skipped=report.skipped,
            indexed=report.indexed,
            error=report.error,
        )
        return report

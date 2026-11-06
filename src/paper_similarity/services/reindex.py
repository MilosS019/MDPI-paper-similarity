"""Rebuild the database and the vector index from the raw landing zone, without arXiv.

Use cases: switching the embedding model (new vector size), restoring a lost database or
index, or replaying data after a parser fix.

Note: the collection is dropped and rebuilt in place, so similarity results are incomplete
while the command runs. In production a new collection would be built next to the old one and
switched over atomically with a Qdrant collection alias.
"""

from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paper_similarity.db.repositories import PaperRepository
from paper_similarity.ingestion.raw_reader import RawParquetReader
from paper_similarity.services.indexing import IndexingService
from paper_similarity.vectorstore.qdrant_store import QdrantVectorStore

log = structlog.get_logger(__name__)

UPSERT_BATCH = 1000  # stays well under Postgres' 32,767 bind-parameter limit


@dataclass(frozen=True, slots=True)
class ReindexReport:
    raw_papers: int
    inserted: int
    updated: int
    unchanged: int
    indexed: int


class ReindexService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        raw_reader: RawParquetReader,
        indexer: IndexingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._session_factory = session_factory
        self._raw = raw_reader
        self._indexer = indexer
        self._store = vector_store

    async def run(self, *, from_raw: bool = True) -> ReindexReport:
        inserted = updated = unchanged = 0
        papers = self._raw.latest_papers() if from_raw else []
        log.info("reindex_started", from_raw=from_raw, raw_papers=len(papers))

        for start in range(0, len(papers), UPSERT_BATCH):
            async with self._session_factory() as session, session.begin():
                result = await PaperRepository(session).upsert(papers[start : start + UPSERT_BATCH])
            inserted += result.inserted
            updated += result.updated
            unchanged += result.unchanged

        await self._store.recreate()
        async with self._session_factory() as session, session.begin():
            reset = await PaperRepository(session).reset_index_state()
        log.info("reindex_index_reset", papers=reset)

        indexed = await self._indexer.index_pending()
        report = ReindexReport(
            raw_papers=len(papers),
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
            indexed=indexed,
        )
        log.info(
            "reindex_finished",
            raw_papers=report.raw_papers,
            inserted=report.inserted,
            updated=report.updated,
            unchanged=report.unchanged,
            indexed=report.indexed,
        )
        return report

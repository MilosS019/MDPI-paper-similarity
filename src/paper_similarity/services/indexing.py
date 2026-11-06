"""Bring the vector index up to date with Postgres.

This is the only place where data is written to both stores. The order makes partial
failures safe to retry:

1. papers are already in Postgres with ``indexed_at = NULL`` (written by the upsert);
2. their vectors are written to Qdrant (idempotent: deterministic point ids);
3. only then are the rows marked ``indexed_at = now()``.

If step 2 or 3 fails, the rows stay pending and the next run picks them up again. The worst
case is a point in Qdrant whose row is not marked yet, which the next run simply overwrites.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paper_similarity.db.repositories import PaperRepository
from paper_similarity.embeddings.base import Embedder, embed_in_thread
from paper_similarity.vectorstore.qdrant_store import PaperVector, QdrantVectorStore

log = structlog.get_logger(__name__)


class IndexingService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        batch_size: int = 64,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._store = vector_store
        self._batch_size = batch_size

    async def index_pending(self) -> int:
        """Embed and index every paper whose vector is missing or stale. Returns the count."""
        indexed = 0
        after_id = 0
        while True:
            async with self._session_factory() as session:
                batch = await PaperRepository(session).pending(
                    self._embedder.model_name, self._batch_size, after_id=after_id
                )
            if not batch:
                break
            after_id = batch[-1].id

            vectors = await embed_in_thread(
                self._embedder, [p.embedding_text for p in batch], purpose="paper"
            )
            await self._store.upsert(
                [
                    PaperVector(arxiv_id=p.arxiv_id, title=p.title, vector=v)
                    for p, v in zip(batch, vectors, strict=True)
                ]
            )
            async with self._session_factory() as session, session.begin():
                await PaperRepository(session).mark_indexed(batch, self._embedder.model_name)

            indexed += len(batch)
            log.info("papers_indexed", batch=len(batch), total=indexed)
        return indexed

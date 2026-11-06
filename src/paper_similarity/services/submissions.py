"""Submitting a manuscript and finding the indexed papers most similar to it."""

import uuid
from datetime import UTC, datetime

import anyio.to_thread
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from paper_similarity.core import metrics
from paper_similarity.db.repositories import PaperRepository, SubmissionRepository
from paper_similarity.domain.errors import EmptyTextError, SubmissionNotFoundError
from paper_similarity.domain.models import EmbeddedChunk, SimilarMatch, Submission, TextChunk
from paper_similarity.domain.text import build_embedding_text
from paper_similarity.embeddings.base import Embedder, embed_in_thread
from paper_similarity.embeddings.chunking import chunk_text
from paper_similarity.services.ranking import aggregate_max
from paper_similarity.vectorstore.qdrant_store import QdrantVectorStore

log = structlog.get_logger(__name__)

# Extra results requested per chunk, so a few hits whose Postgres row is missing (possible
# only after an interrupted dual write) can be dropped without returning fewer than `limit`.
ORPHAN_BUFFER = 5


class SubmissionService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        *,
        chunk_max_tokens: int,
        chunk_overlap_tokens: int,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._store = vector_store
        self._chunk_max_tokens = chunk_max_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens

    async def create(self, title: str | None, text: str) -> Submission:
        """Embed and persist a submission in one database transaction.

        The submission's vectors live in Postgres, not in Qdrant: they are only ever read back
        by id, never searched. That keeps this write atomic, with no second store to keep in
        sync on the request path.
        """
        chunks = await self._embed(title, text)
        submission = Submission(
            id=uuid.uuid4(),
            title=title,
            text=text,
            embedding_model=self._embedder.model_name,
            created_at=datetime.now(UTC),
            chunks=tuple(chunks),
        )
        async with self._session_factory() as session, session.begin():
            await SubmissionRepository(session).add(submission)
        log.info("submission_created", submission_id=str(submission.id), chunks=len(chunks))
        return submission

    async def get(self, submission_id: uuid.UUID) -> Submission:
        async with self._session_factory() as session:
            submission = await SubmissionRepository(session).get(submission_id)
        if submission is None:
            raise SubmissionNotFoundError(submission_id)
        return submission

    async def find_similar(
        self, submission_id: uuid.UUID, limit: int
    ) -> tuple[Submission, list[SimilarMatch]]:
        submission = await self.get(submission_id)
        vectors = [c.vector for c in submission.chunks]
        if submission.embedding_model != self._embedder.model_name:
            # Stored vectors came from another model and are not comparable with the index.
            # The text is stored, so re-embed it with the current model instead of failing.
            log.info(
                "submission_reembedded",
                submission_id=str(submission_id),
                stored_model=submission.embedding_model,
                current_model=self._embedder.model_name,
            )
            vectors = [c.vector for c in await self._embed(submission.title, submission.text)]

        hits_per_chunk = await self._store.search_batch(vectors, limit + ORPHAN_BUFFER)
        ranked = aggregate_max(hits_per_chunk)

        async with self._session_factory() as session:
            papers = await PaperRepository(session).get_many(h.arxiv_id for h in ranked)

        matches: list[SimilarMatch] = []
        for hit in ranked:
            paper = papers.get(hit.arxiv_id)
            if paper is None:
                log.warning("similar_hit_without_row", arxiv_id=hit.arxiv_id)
                continue
            matches.append(
                SimilarMatch(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    url=paper.abs_url,
                    published=paper.published,
                    primary_category=paper.primary_category,
                    score=hit.score,
                    matched_chunk=hit.chunk_index,
                )
            )
            if len(matches) == limit:
                break
        metrics.SIMILARITY_RESULTS.observe(len(matches))
        return submission, matches

    async def _embed(self, title: str | None, text: str) -> list[EmbeddedChunk]:
        source = build_embedding_text(title, text)
        if not source:
            raise EmptyTextError("text must contain at least one non-whitespace character")
        # Tokenising a long manuscript is CPU work too, so it also runs off the event loop.
        chunks: list[TextChunk] = await anyio.to_thread.run_sync(
            lambda: chunk_text(
                source,
                self._embedder.token_offsets,
                max_tokens=self._chunk_max_tokens,
                overlap_tokens=self._chunk_overlap_tokens,
            )
        )
        vectors = await embed_in_thread(
            self._embedder, [c.text for c in chunks], purpose="submission_chunk"
        )
        return [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors, strict=True)]

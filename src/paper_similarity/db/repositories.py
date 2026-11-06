"""Data access for papers and submissions.

Repositories translate between ORM rows and domain objects and contain the SQL. They never
commit: the calling service owns the transaction boundary.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    ColumnElement,
    and_,
    case,
    delete,
    func,
    literal_column,
    null,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from paper_similarity.db.models import ArxivPaperRow, SubmissionChunkRow, SubmissionRow
from paper_similarity.domain.models import ArxivPaper, EmbeddedChunk, Submission, TextChunk
from paper_similarity.domain.text import build_embedding_text, content_hash


@dataclass(frozen=True, slots=True)
class UpsertResult:
    inserted: int
    updated: int
    unchanged: int


@dataclass(frozen=True, slots=True)
class PendingPaper:
    """A paper whose vector is missing or stale, with the text that should be embedded."""

    id: int
    arxiv_id: str
    title: str
    embedding_text: str
    content_hash: str


# Every column the upsert writes from arXiv data.
_COMPARED = (
    "version",
    "title",
    "abstract",
    "authors",
    "categories",
    "primary_category",
    "published",
    "source_updated",
    "abs_url",
    "pdf_url",
    "content_hash",
)


def _row_to_paper(row: ArxivPaperRow) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=row.arxiv_id,
        version=row.version,
        title=row.title,
        abstract=row.abstract,
        authors=tuple(row.authors),
        categories=tuple(row.categories),
        primary_category=row.primary_category,
        published=row.published,
        updated=row.source_updated,
        abs_url=row.abs_url,
        pdf_url=row.pdf_url,
    )


def _latest_per_id(papers: Iterable[ArxivPaper]) -> list[ArxivPaper]:
    """Keep one record per arxiv_id (the most recent). Postgres rejects an upsert batch that
    touches the same row twice, and arXiv paging can return the same paper on two pages."""
    latest: dict[str, ArxivPaper] = {}
    for paper in papers:
        current = latest.get(paper.arxiv_id)
        if current is None or (paper.version, paper.updated) > (current.version, current.updated):
            latest[paper.arxiv_id] = paper
    return list(latest.values())


class PaperRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, papers: Sequence[ArxivPaper]) -> UpsertResult:
        """Insert new papers and update changed ones in a single statement.

        * A paper is matched on ``arxiv_id`` (version-less), so re-ingesting never duplicates.
        * Older data never overwrites newer data (``source_updated`` guard), which matters when
          replaying old Parquet files.
        * If the embedded text changed, ``indexed_at`` is reset so the vector is rebuilt;
          metadata-only changes keep the existing vector.
        """
        unique = _latest_per_id(papers)
        if not unique:
            return UpsertResult(inserted=0, updated=0, unchanged=0)

        values = []
        for p in unique:
            values.append(
                {
                    "arxiv_id": p.arxiv_id,
                    "version": p.version,
                    "title": p.title,
                    "abstract": p.abstract,
                    "authors": list(p.authors),
                    "categories": list(p.categories),
                    "primary_category": p.primary_category,
                    "published": p.published,
                    "source_updated": p.updated,
                    "abs_url": p.abs_url,
                    "pdf_url": p.pdf_url,
                    "content_hash": content_hash(build_embedding_text(p.title, p.abstract)),
                }
            )

        base = insert(ArxivPaperRow).values(values)
        excluded = base.excluded
        table = ArxivPaperRow.__table__.c
        text_unchanged = table.content_hash == excluded.content_hash
        upsert = base.on_conflict_do_update(
            index_elements=[ArxivPaperRow.arxiv_id],
            set_={
                "version": excluded.version,
                "title": excluded.title,
                "abstract": excluded.abstract,
                "authors": excluded.authors,
                "categories": excluded.categories,
                "primary_category": excluded.primary_category,
                "published": excluded.published,
                "source_updated": excluded.source_updated,
                "abs_url": excluded.abs_url,
                "pdf_url": excluded.pdf_url,
                "content_hash": excluded.content_hash,
                "indexed_at": _keep_if(text_unchanged, table.indexed_at),
                "embedding_model": _keep_if(text_unchanged, table.embedding_model),
                "updated_at": func.now(),
            },
            # Update only with data at least as new, and only if some stored field differs.
            # Comparing every field (not just the timestamp) catches metadata changes arXiv
            # makes without a new <updated> time, e.g. a paper cross-listed to a new category.
            where=and_(
                excluded.source_updated >= table.source_updated,
                or_(*(table[name].is_distinct_from(excluded[name]) for name in _COMPARED)),
            ),
        )
        # xmax = 0 only for freshly inserted tuples: a well-known Postgres idiom to tell
        # inserts from updates in an upsert. Skipped (unchanged) rows return nothing.
        returning = upsert.returning(literal_column("(xmax = 0)", type_=Boolean))
        flags = [bool(flag) for flag in (await self._session.scalars(returning)).all()]
        inserted = sum(flags)
        updated = len(flags) - inserted
        return UpsertResult(inserted=inserted, updated=updated, unchanged=len(unique) - len(flags))

    async def pending(self, limit: int, *, after_id: int = 0) -> list[PendingPaper]:
        """Papers whose vector is missing or stale (``indexed_at IS NULL``).

        Served by the partial index ``ix_arxiv_papers_pending``, so the cost depends on the
        number of pending rows, not the table size. A model change does not need a separate
        condition here: the vector store refuses a collection built by another model, and
        ``pulse reindex`` resets ``indexed_at`` for every row.

        Keyset pagination (``id > after_id``) guarantees a single pass over the rows even if
        some of them stay pending, so callers can loop until an empty batch.
        """
        stmt = (
            select(ArxivPaperRow)
            .where(ArxivPaperRow.id > after_id, ArxivPaperRow.indexed_at.is_(None))
            .order_by(ArxivPaperRow.id)
            .limit(limit)
        )
        rows = (await self._session.scalars(stmt)).all()
        result = []
        for row in rows:
            text = build_embedding_text(row.title, row.abstract)
            result.append(
                PendingPaper(
                    id=row.id,
                    arxiv_id=row.arxiv_id,
                    title=row.title,
                    embedding_text=text,
                    content_hash=row.content_hash,
                )
            )
        return result

    async def mark_indexed(self, items: Sequence[PendingPaper], embedding_model: str) -> None:
        """Mark papers as indexed, but only if their text did not change in the meantime.

        One statement for the whole batch: ``WHERE (arxiv_id, content_hash) IN (...)``.
        """
        if not items:
            return
        await self._session.execute(
            update(ArxivPaperRow)
            .where(
                tuple_(ArxivPaperRow.arxiv_id, ArxivPaperRow.content_hash).in_(
                    [(item.arxiv_id, item.content_hash) for item in items]
                )
            )
            .values(indexed_at=datetime.now(UTC), embedding_model=embedding_model)
        )

    async def reset_index_state(self) -> int:
        """Mark every paper as needing a new vector (used by a full reindex)."""
        result = await self._session.execute(
            update(ArxivPaperRow).values(indexed_at=None, embedding_model=None)
        )
        return int(result.rowcount)  # type: ignore[attr-defined]

    async def get_many(self, arxiv_ids: Iterable[str]) -> dict[str, ArxivPaper]:
        ids = list(set(arxiv_ids))
        if not ids:
            return {}
        rows = await self._session.scalars(
            select(ArxivPaperRow).where(ArxivPaperRow.arxiv_id.in_(ids))
        )
        return {row.arxiv_id: _row_to_paper(row) for row in rows}

    async def counts(self, embedding_model: str) -> tuple[int, int]:
        """Return ``(total papers, papers with an up-to-date vector from this model)``."""
        indexed_now = and_(
            ArxivPaperRow.indexed_at.is_not(None),
            ArxivPaperRow.embedding_model == embedding_model,
        )
        total, indexed = (
            await self._session.execute(
                select(func.count(), func.count().filter(indexed_now)).select_from(ArxivPaperRow)
            )
        ).one()
        return int(total), int(indexed)


class SubmissionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, submission: Submission) -> None:
        row = SubmissionRow(
            id=submission.id,
            title=submission.title,
            text=submission.text,
            embedding_model=submission.embedding_model,
            created_at=submission.created_at,
            chunks=[
                SubmissionChunkRow(
                    chunk_index=c.chunk.index,
                    char_start=c.chunk.char_start,
                    char_end=c.chunk.char_end,
                    vector=c.vector,
                )
                for c in submission.chunks
            ],
        )
        self._session.add(row)
        await self._session.flush()

    async def replace_embedding(
        self, submission_id: UUID, embedding_model: str, chunks: Sequence[EmbeddedChunk]
    ) -> bool:
        """Replace a submission's chunk vectors with ones from ``embedding_model``.

        Locks the submission row first, so concurrent requests re-embedding the same
        submission do the swap once. Returns False if it was already up to date (or gone).
        """
        current = await self._session.scalar(
            select(SubmissionRow.embedding_model)
            .where(SubmissionRow.id == submission_id)
            .with_for_update()
        )
        if current is None or current == embedding_model:
            return False
        await self._session.execute(
            delete(SubmissionChunkRow).where(SubmissionChunkRow.submission_id == submission_id)
        )
        self._session.add_all(
            SubmissionChunkRow(
                submission_id=submission_id,
                chunk_index=c.chunk.index,
                char_start=c.chunk.char_start,
                char_end=c.chunk.char_end,
                vector=c.vector,
            )
            for c in chunks
        )
        await self._session.execute(
            update(SubmissionRow)
            .where(SubmissionRow.id == submission_id)
            .values(embedding_model=embedding_model)
        )
        await self._session.flush()
        return True

    async def get(self, submission_id: UUID) -> Submission | None:
        row = await self._session.get(SubmissionRow, submission_id)
        if row is None:
            return None
        # Chunk offsets index into the embedding text, which is derived deterministically
        # from the stored title and text.
        source = build_embedding_text(row.title, row.text)
        return Submission(
            id=row.id,
            title=row.title,
            text=row.text,
            embedding_model=row.embedding_model,
            created_at=row.created_at,
            chunks=tuple(
                EmbeddedChunk(
                    chunk=TextChunk(
                        index=c.chunk_index,
                        text=source[c.char_start : c.char_end],
                        char_start=c.char_start,
                        char_end=c.char_end,
                    ),
                    vector=list(c.vector),
                )
                for c in row.chunks
            ),
        )


def _keep_if(
    condition: ColumnElement[bool], column: ColumnElement[object]
) -> ColumnElement[object]:
    """``CASE WHEN condition THEN column ELSE NULL END``."""
    return case((condition, column), else_=null())

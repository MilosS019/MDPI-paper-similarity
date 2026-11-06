"""SQLAlchemy table definitions. Postgres is the source of truth; Qdrant is a derived index."""

import uuid
from datetime import datetime

from sqlalchemy import (
    REAL,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names, so Alembic migrations stay stable and readable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ArxivPaperRow(Base):
    __tablename__ = "arxiv_papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True)
    version: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(ARRAY(Text))
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text))
    primary_category: Mapped[str | None] = mapped_column(String(64))
    published: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    abs_url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    # SHA-256 of the exact text that was (or will be) embedded.
    content_hash: Mapped[str] = mapped_column(String(64))
    # Set only after the vector is confirmed in Qdrant; NULL means "needs indexing".
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # Fast lookup of rows still waiting for their vector.
        Index(
            "ix_arxiv_papers_pending",
            "id",
            postgresql_where="indexed_at IS NULL",
        ),
    )


class SubmissionRow(Base):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["SubmissionChunkRow"]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        order_by="SubmissionChunkRow.chunk_index",
        lazy="selectin",
    )


class SubmissionChunkRow(Base):
    """One embedded chunk of a submission. Stored here, not in Qdrant: see docs/architecture.md."""

    __tablename__ = "submission_chunks"

    submission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list[float]] = mapped_column(ARRAY(REAL))

    submission: Mapped[SubmissionRow] = relationship(back_populates="chunks")

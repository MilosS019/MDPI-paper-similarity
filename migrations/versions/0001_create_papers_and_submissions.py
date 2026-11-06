"""create papers and submissions

Revision ID: a6452600e9a5
Revises:
Create Date: 2026-11-06 12:01:51.573600
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a6452600e9a5"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "arxiv_papers",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("arxiv_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("authors", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("primary_category", sa.String(length=64), nullable=True),
        sa.Column("published", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("abs_url", sa.Text(), nullable=False),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_arxiv_papers")),
        sa.UniqueConstraint("arxiv_id", name=op.f("uq_arxiv_papers_arxiv_id")),
    )
    op.create_index(
        "ix_arxiv_papers_pending",
        "arxiv_papers",
        ["id"],
        unique=False,
        postgresql_where="indexed_at IS NULL",
    )
    op.create_table(
        "submissions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_submissions")),
    )
    op.create_table(
        "submission_chunks",
        sa.Column("submission_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("vector", postgresql.ARRAY(sa.REAL()), nullable=False),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submissions.id"],
            name=op.f("fk_submission_chunks_submission_id_submissions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("submission_id", "chunk_index", name=op.f("pk_submission_chunks")),
    )


def downgrade() -> None:
    op.drop_table("submission_chunks")
    op.drop_table("submissions")
    op.drop_index(
        "ix_arxiv_papers_pending", table_name="arxiv_papers", postgresql_where="indexed_at IS NULL"
    )
    op.drop_table("arxiv_papers")

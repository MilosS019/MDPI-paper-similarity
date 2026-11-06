"""Domain objects shared across layers.

These are plain, immutable dataclasses with no dependency on the web framework, the ORM or
the vector store, so every layer can use them and they are trivial to build in tests.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ArxivPaper:
    """One arXiv paper as parsed from the API. ``arxiv_id`` never carries a version suffix."""

    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str | None
    published: datetime
    updated: datetime
    abs_url: str
    pdf_url: str | None


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A window of a longer text. ``char_start``/``char_end`` index into the source text."""

    index: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: TextChunk
    vector: list[float]


@dataclass(frozen=True, slots=True)
class Submission:
    id: UUID
    title: str | None
    text: str
    embedding_model: str
    created_at: datetime
    chunks: tuple[EmbeddedChunk, ...]


@dataclass(frozen=True, slots=True)
class VectorHit:
    """A raw hit from the vector store: which indexed paper, and its cosine similarity."""

    arxiv_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """A vector hit together with the submission chunk whose query produced it."""

    arxiv_id: str
    score: float
    chunk_index: int


@dataclass(frozen=True, slots=True)
class SimilarMatch:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    url: str
    published: datetime
    primary_category: str | None
    score: float
    matched_chunk: int

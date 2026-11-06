"""Builds the long-lived resources (pools, clients, model) from settings.

Shared by the API (in its lifespan) and the CLI, so both are wired identically. Tests pass
their own embedder to avoid loading the real model.
"""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from paper_similarity.core.config import Settings
from paper_similarity.db.session import create_engine, create_session_factory
from paper_similarity.embeddings.base import Embedder
from paper_similarity.vectorstore.qdrant_store import QdrantVectorStore


@dataclass(slots=True)
class Resources:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    qdrant: AsyncQdrantClient
    vector_store: QdrantVectorStore
    embedder: Embedder

    async def close(self) -> None:
        await self.qdrant.close()
        await self.engine.dispose()


def load_embedder(settings: Settings) -> Embedder:
    # Imported lazily: loading ONNX Runtime and the model takes a moment and is not needed
    # by every caller (e.g. tests that inject a fake).
    from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder

    return FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        max_tokens=settings.chunk_max_tokens,
    )


def create_resources(settings: Settings, embedder: Embedder | None = None) -> Resources:
    engine = create_engine(settings.database_url)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    resolved_embedder = embedder or load_embedder(settings)
    return Resources(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        qdrant=qdrant,
        vector_store=QdrantVectorStore(
            qdrant,
            settings.qdrant_collection,
            resolved_embedder.dimension,
            resolved_embedder.model_name,
        ),
        embedder=resolved_embedder,
    )

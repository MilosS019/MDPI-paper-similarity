"""Qdrant access: the vector index over arXiv papers.

Qdrant is a derived index: Postgres is the source of truth and the index can always be rebuilt
(``pulse reindex``). Point ids are UUID5 of the arXiv id, so re-indexing a paper overwrites its
point instead of adding a duplicate.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    QueryRequest,
    VectorParams,
)

from paper_similarity.domain.errors import DependencyUnavailableError, PaperSimilarityError
from paper_similarity.domain.models import VectorHit

log = structlog.get_logger(__name__)

# Fixed namespace: the same arXiv id always maps to the same point id, in every environment.
POINT_NAMESPACE = uuid.UUID("6f1c9a52-2d4e-4f0b-9a57-3c1d8e0b7a10")


def point_id(arxiv_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, arxiv_id))


class VectorDimensionMismatchError(PaperSimilarityError):
    """The collection was built for a different model; run ``pulse reindex --recreate``."""


@dataclass(frozen=True, slots=True)
class PaperVector:
    arxiv_id: str
    title: str
    vector: list[float]


@asynccontextmanager
async def _translate_errors() -> AsyncIterator[None]:
    """Turn transport and server failures into one domain error the API maps to 503."""
    try:
        yield
    except (ResponseHandlingException, ApiException, httpx.HTTPError, OSError) as exc:
        raise DependencyUnavailableError("vector store", str(exc) or type(exc).__name__) from exc


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self) -> None:
        """Create the collection if missing; refuse to use one built for another dimension."""
        async with _translate_errors():
            if await self._client.collection_exists(self._collection):
                info = await self._client.get_collection(self._collection)
                params = info.config.params.vectors
                size = params.size if isinstance(params, VectorParams) else None
                if size != self._dimension:
                    raise VectorDimensionMismatchError(
                        f"collection {self._collection!r} stores {size}-dimensional vectors but "
                        f"the embedding model produces {self._dimension}; "
                        "run `pulse reindex --recreate`"
                    )
                return
            await self._create()

    async def recreate(self) -> None:
        """Drop and recreate the collection (used when the embedding model changes)."""
        async with _translate_errors():
            if await self._client.collection_exists(self._collection):
                await self._client.delete_collection(self._collection)
            await self._create()

    async def _create(self) -> None:
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="arxiv_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("qdrant_collection_created", collection=self._collection, size=self._dimension)

    async def upsert(self, items: Sequence[PaperVector]) -> None:
        if not items:
            return
        points = [
            PointStruct(
                id=point_id(item.arxiv_id),
                vector=item.vector,
                payload={"arxiv_id": item.arxiv_id, "title": item.title},
            )
            for item in items
        ]
        async with _translate_errors():
            # wait=True: return only once the points are persisted and searchable, so the
            # caller can safely mark the papers as indexed in Postgres afterwards.
            await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def search_batch(
        self, vectors: Sequence[Sequence[float]], limit: int
    ) -> list[list[VectorHit]]:
        """Nearest papers for each query vector, in one round trip. Scores are cosine similarity."""
        if not vectors:
            return []
        requests = [
            QueryRequest(query=list(vector), limit=limit, with_payload=["arxiv_id"])
            for vector in vectors
        ]
        async with _translate_errors():
            responses = await self._client.query_batch_points(
                collection_name=self._collection, requests=requests
            )
        results: list[list[VectorHit]] = []
        for response in responses:
            hits = []
            for point in response.points:
                arxiv_id = (point.payload or {}).get("arxiv_id")
                if isinstance(arxiv_id, str):
                    hits.append(VectorHit(arxiv_id=arxiv_id, score=float(point.score)))
            results.append(hits)
        return results

    async def count(self) -> int:
        async with _translate_errors():
            result = await self._client.count(collection_name=self._collection, exact=True)
        return int(result.count)

    async def ping(self) -> None:
        """Raise ``DependencyUnavailableError`` unless Qdrant answers and the collection exists."""
        async with _translate_errors():
            if not await self._client.collection_exists(self._collection):
                raise DependencyUnavailableError(
                    "vector store", f"collection {self._collection!r} does not exist"
                )

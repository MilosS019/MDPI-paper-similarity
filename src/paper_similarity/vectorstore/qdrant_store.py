"""Qdrant access: the vector index over arXiv papers.

Qdrant is a derived index: Postgres is the source of truth and the index can always be rebuilt
(``pulse reindex``). Point ids are UUID5 of the arXiv id, so re-indexing a paper overwrites its
point instead of adding a duplicate.

The collection records the embedding model that built it (collection metadata). Vectors from
different models are not comparable, even at the same dimension, so a collection built by
another model is refused until it is rebuilt.

The collection is checked lazily, before first use, rather than only at startup: if Qdrant
is down when the API starts, the collection is created or checked as soon as it is back.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    QueryRequest,
    VectorParams,
)

from paper_similarity.domain.errors import (
    CollectionMissingError,
    DependencyUnavailableError,
    IndexModelMismatchError,
    VectorStoreError,
)
from paper_similarity.domain.models import VectorHit

log = structlog.get_logger(__name__)

# Fixed namespace: the same arXiv id always maps to the same point id, in every environment.
POINT_NAMESPACE = uuid.UUID("6f1c9a52-2d4e-4f0b-9a57-3c1d8e0b7a10")
MODEL_METADATA_KEY = "embedding_model"


def point_id(arxiv_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, arxiv_id))


@dataclass(frozen=True, slots=True)
class PaperVector:
    arxiv_id: str
    title: str
    vector: list[float]


class QdrantVectorStore:
    def __init__(
        self, client: AsyncQdrantClient, collection: str, dimension: int, model_name: str
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension
        self._model_name = model_name
        self._verified = False

    @property
    def collection(self) -> str:
        return self._collection

    @asynccontextmanager
    async def _translate_errors(self) -> AsyncIterator[None]:
        """Map client errors to domain errors.

        * Transport failures and 5xx responses are temporary: ``DependencyUnavailableError``
          (the API answers 503 with Retry-After).
        * 4xx responses are permanent (wrong request, missing collection): ``VectorStoreError``.
          A 404 means the collection is gone (e.g. wiped Qdrant storage):
          ``CollectionMissingError``, which ``_with_collection`` handles by recreating it.
        """
        try:
            yield
        except UnexpectedResponse as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                raise DependencyUnavailableError("vector store", str(exc)) from exc
            if exc.status_code == 404:
                self._verified = False
                raise CollectionMissingError(f"collection {self._collection!r} not found") from exc
            raise VectorStoreError(f"Qdrant rejected the request: {exc}") from exc
        except (ResponseHandlingException, httpx.HTTPError, OSError) as exc:
            raise DependencyUnavailableError(
                "vector store", str(exc) or type(exc).__name__
            ) from exc

    async def ensure_collection(self) -> None:
        """Create the collection if missing; refuse one built by another model or dimension."""
        async with self._translate_errors():
            if not await self._client.collection_exists(self._collection):
                await self._create()
            else:
                info = await self._client.get_collection(self._collection)
                params = info.config.params.vectors
                size = params.size if isinstance(params, VectorParams) else None
                built_with = (info.config.metadata or {}).get(MODEL_METADATA_KEY)
                if size != self._dimension or built_with != self._model_name:
                    raise IndexModelMismatchError(
                        f"collection {self._collection!r} was built with model {built_with!r} "
                        f"({size} dimensions) but the configured model is "
                        f"{self._model_name!r} ({self._dimension} dimensions); "
                        "run `pulse reindex` to rebuild it"
                    )
        self._verified = True

    async def _with_collection[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run an operation once the collection is verified; if the collection vanished in the
        meantime, recreate it and retry once (the index is then rebuilt by `pulse index`)."""
        if not self._verified:
            await self.ensure_collection()
        try:
            async with self._translate_errors():
                return await operation()
        except CollectionMissingError:
            log.warning(
                "qdrant_collection_missing_recreating",
                collection=self._collection,
                hint="the index is now empty; run `pulse reindex --from-db` to rebuild it",
            )
            await self.ensure_collection()
            async with self._translate_errors():
                return await operation()

    async def recreate(self) -> None:
        """Drop and recreate the collection (used when the embedding model changes)."""
        async with self._translate_errors():
            if await self._client.collection_exists(self._collection):
                await self._client.delete_collection(self._collection)
            await self._create()
        self._verified = True

    async def _create(self) -> None:
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            metadata={MODEL_METADATA_KEY: self._model_name},
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="arxiv_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info(
            "qdrant_collection_created",
            collection=self._collection,
            size=self._dimension,
            model=self._model_name,
        )

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
        # wait=True: return only once the points are persisted and searchable, so the caller
        # can safely mark the papers as indexed in Postgres afterwards.
        await self._with_collection(
            lambda: self._client.upsert(collection_name=self._collection, points=points, wait=True)
        )

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
        responses = await self._with_collection(
            lambda: self._client.query_batch_points(
                collection_name=self._collection, requests=requests
            )
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
        result = await self._with_collection(
            lambda: self._client.count(collection_name=self._collection, exact=True)
        )
        return int(result.count)

    async def ping(self) -> None:
        """Raise unless Qdrant answers and the collection exists and matches the model."""
        await self.ensure_collection()

"""Vector store behaviour against Qdrant's in-process (":memory:") engine: same client API,
no server needed. The integration suite repeats the core flow against a real Qdrant."""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from paper_similarity.domain.errors import IndexModelMismatchError
from paper_similarity.vectorstore.qdrant_store import PaperVector, QdrantVectorStore, point_id
from tests.fakes import HashingEmbedder

embedder = HashingEmbedder()
MODEL = embedder.model_name


@pytest.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(location=":memory:")
    yield qdrant
    await qdrant.close()


@pytest.fixture
async def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    vector_store = QdrantVectorStore(client, "papers", embedder.dimension, MODEL)
    await vector_store.ensure_collection()
    return vector_store


def paper(arxiv_id: str, text: str) -> PaperVector:
    return PaperVector(arxiv_id=arxiv_id, title=text, vector=embedder.embed([text])[0])


def test_point_id_is_deterministic() -> None:
    assert point_id("2401.12345") == point_id("2401.12345")
    assert point_id("2401.12345") != point_id("2401.12346")


async def test_ensure_collection_is_idempotent(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    assert await store.count() == 0


async def test_collection_from_another_dimension_is_refused(
    client: AsyncQdrantClient, store: QdrantVectorStore
) -> None:
    other = QdrantVectorStore(client, "papers", embedder.dimension + 1, MODEL)
    with pytest.raises(IndexModelMismatchError, match="pulse reindex"):
        await other.ensure_collection()


async def test_collection_from_another_model_with_same_dimension_is_refused(
    client: AsyncQdrantClient, store: QdrantVectorStore
) -> None:
    other = QdrantVectorStore(client, "papers", embedder.dimension, "other/model-384")
    with pytest.raises(IndexModelMismatchError, match="other/model-384"):
        await other.search_batch([[0.0] * embedder.dimension], limit=1)


async def test_collection_is_created_lazily_on_first_use(client: AsyncQdrantClient) -> None:
    lazy = QdrantVectorStore(client, "lazy", embedder.dimension, MODEL)  # never "started"
    await lazy.upsert([paper("1", "graph neural networks")])
    assert await lazy.count() == 1


async def test_upserting_the_same_paper_twice_keeps_one_point(store: QdrantVectorStore) -> None:
    await store.upsert([paper("1", "graph neural networks")])
    await store.upsert([paper("1", "graph neural networks for molecules")])
    assert await store.count() == 1


async def test_search_batch_ranks_by_cosine_similarity(store: QdrantVectorStore) -> None:
    await store.upsert(
        [
            paper("bio", "protein folding in yeast cells"),
            paper("ir", "dense retrieval of scientific papers with transformers"),
            paper("cv", "image segmentation with convolutional networks"),
        ]
    )
    queries = embedder.embed(["transformers for retrieval of papers", "yeast protein folding"])

    first, second = await store.search_batch(queries, limit=3)

    assert first[0].arxiv_id == "ir"
    assert second[0].arxiv_id == "bio"
    assert [h.score for h in first] == sorted((h.score for h in first), reverse=True)
    assert -1.0 <= first[-1].score <= first[0].score <= 1.0


async def test_recreate_empties_the_collection(store: QdrantVectorStore) -> None:
    await store.upsert([paper("1", "anything")])
    await store.recreate()
    assert await store.count() == 0

"""API behaviour end to end: HTTP -> services -> real Postgres and Qdrant (fake embedder)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy import update
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from paper_similarity.api.app import create_app
from paper_similarity.bootstrap import Resources, create_resources
from paper_similarity.core.config import Settings
from paper_similarity.db.models import SubmissionRow
from paper_similarity.db.repositories import PaperRepository
from paper_similarity.domain.errors import IndexModelMismatchError
from paper_similarity.domain.models import ArxivPaper
from paper_similarity.services.indexing import IndexingService
from paper_similarity.services.submissions import SubmissionService
from tests.fakes import HashingEmbedder

CORPUS = {
    "2401.00001": (
        "Dense retrieval for scientific question answering",
        "We train dual encoder transformers to retrieve passages from scientific papers "
        "and answer domain questions with retrieval augmented generation.",
    ),
    "2401.00002": (
        "Protein folding dynamics in yeast",
        "Molecular dynamics simulations reveal protein folding pathways in yeast cells "
        "under heat stress conditions.",
    ),
    "2401.00003": (
        "Galaxy cluster lensing",
        "Weak gravitational lensing measurements constrain the dark matter mass of galaxy "
        "clusters observed by wide field surveys.",
    ),
    "2401.00004": (
        "Graph neural networks for molecules",
        "Message passing graph neural networks predict molecular properties and protein "
        "binding affinities from atomic graphs.",
    ),
}


def arxiv_paper(arxiv_id: str, title: str, abstract: str) -> ArxivPaper:
    when = datetime(2024, 1, 1, tzinfo=UTC)
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=1,
        title=title,
        abstract=abstract,
        authors=("Ada Lovelace", "Alan Turing"),
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=when,
        updated=when,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=None,
    )


@pytest.fixture
async def indexed_corpus(resources: Resources) -> None:
    async with resources.session_factory() as session, session.begin():
        await PaperRepository(session).upsert(
            [arxiv_paper(i, t, a) for i, (t, a) in CORPUS.items()]
        )
    indexer = IndexingService(resources.session_factory, resources.embedder, resources.vector_store)
    assert await indexer.index_pending() == len(CORPUS)


@asynccontextmanager
async def running_app(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(settings, lambda s: create_resources(s, embedder=HashingEmbedder()))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def client(settings: Settings, resources: Resources) -> AsyncIterator[httpx.AsyncClient]:
    # `resources` (from conftest) resets the database and collection before each test.
    async with running_app(settings) as api:
        yield api


async def submit(client: httpx.AsyncClient, text: str, title: str | None = None) -> UUID:
    response = await client.post("/api/v1/papers", json={"title": title, "text": text})
    assert response.status_code == 201, response.text
    return UUID(response.json()["id"])


def assert_problem(response: httpx.Response, status: int) -> dict[str, object]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/problem+json")
    body: dict[str, object] = response.json()
    assert body["status"] == status
    assert body["request_id"] == response.headers["x-request-id"]
    return body


# --- POST /api/v1/papers ---------------------------------------------------------------


async def test_submit_paper_returns_201_with_id_and_location(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/papers", json={"title": "My manuscript", "text": "Some abstract text."}
    )

    assert response.status_code == 201
    body = response.json()
    paper_id = UUID(body["id"])
    assert body["status"] == "created"
    assert body["chunks"] == 1
    assert body["embedding_model"] == HashingEmbedder.model_name
    assert response.headers["location"] == f"/api/v1/papers/{paper_id}"

    stored = (await client.get(f"/api/v1/papers/{paper_id}")).json()
    assert stored["title"] == "My manuscript"
    assert stored["text_length"] == len("Some abstract text.")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"title": "only a title"},
        {"text": ""},
        {"text": "   \n\t "},
        {"text": 42},
        {"text": "ok", "unexpected": True},
    ],
)
async def test_submit_paper_rejects_invalid_bodies(
    client: httpx.AsyncClient, payload: dict[str, object]
) -> None:
    body = assert_problem(await client.post("/api/v1/papers", json=payload), 422)
    assert body["errors"]


async def test_submit_paper_rejects_malformed_json(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/papers", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert_problem(response, 422)


async def test_submit_paper_rejects_text_over_the_limit(
    settings: Settings, resources: Resources
) -> None:
    async with running_app(settings.model_copy(update={"max_submission_chars": 50})) as api:
        body = assert_problem(await api.post("/api/v1/papers", json={"text": "x" * 51}), 422)
    assert "maximum is 50" in str(body["detail"])


async def test_long_submission_is_chunked(settings: Settings, resources: Resources) -> None:
    small_chunks = settings.model_copy(update={"chunk_max_tokens": 34, "chunk_overlap_tokens": 4})
    text = " ".join(f"word{i}" for i in range(200))
    async with running_app(small_chunks) as api:
        response = await api.post("/api/v1/papers", json={"text": text})
    assert response.status_code == 201
    assert response.json()["chunks"] == 7  # 32 content tokens per window, step 28


# --- GET /api/v1/posts/{id}/similar ---------------------------------------------------


@pytest.mark.usefixtures("indexed_corpus")
async def test_similar_papers_are_ranked_by_score(client: httpx.AsyncClient) -> None:
    paper_id = await submit(
        client,
        title="Retrieval augmented question answering",
        text="Transformers retrieve passages from scientific papers to answer questions.",
    )

    response = await client.get(f"/api/v1/posts/{paper_id}/similar", params={"limit": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == str(paper_id)
    assert body["limit"] == 3
    matches = body["matches"]
    assert len(matches) == 3
    assert matches[0]["arxiv_id"] == "2401.00001"
    assert matches[0]["title"] == CORPUS["2401.00001"][0]
    assert matches[0]["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert matches[0]["url"] == "https://arxiv.org/abs/2401.00001"
    assert matches[0]["matched_chunk"] == 0
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(-1 <= s <= 1 for s in scores)


@pytest.mark.usefixtures("indexed_corpus")
async def test_default_limit_and_alias_path(client: httpx.AsyncClient) -> None:
    paper_id = await submit(client, "protein folding in yeast cells")

    canonical = (await client.get(f"/api/v1/posts/{paper_id}/similar")).json()
    alias = (await client.get(f"/api/v1/papers/{paper_id}/similar")).json()

    assert canonical["limit"] == 10
    assert len(canonical["matches"]) == len(CORPUS)  # fewer papers than the limit
    assert canonical["matches"][0]["arxiv_id"] == "2401.00002"
    assert canonical == alias


@pytest.mark.usefixtures("indexed_corpus")
async def test_best_chunk_wins_for_multi_topic_manuscripts(
    settings: Settings, resources: Resources
) -> None:
    # Two topics far apart in one manuscript: with max-score aggregation each topic's paper
    # ranks at the top, matched by a different chunk.
    filler = " ".join(f"filler{i}" for i in range(40))
    text = (
        "weak gravitational lensing constrains dark matter mass of galaxy clusters "
        f"{filler} protein folding pathways in yeast cells under heat stress"
    )
    small_chunks = settings.model_copy(update={"chunk_max_tokens": 22, "chunk_overlap_tokens": 0})
    async with running_app(small_chunks) as api:
        paper_id = await submit(api, text)
        matches = (await api.get(f"/api/v1/posts/{paper_id}/similar?limit=2")).json()["matches"]

    assert {m["arxiv_id"] for m in matches} == {"2401.00002", "2401.00003"}
    assert len({m["matched_chunk"] for m in matches}) == 2


async def test_similar_with_empty_index_returns_no_matches(client: httpx.AsyncClient) -> None:
    paper_id = await submit(client, "anything at all")
    response = await client.get(f"/api/v1/posts/{paper_id}/similar")
    assert response.status_code == 200
    assert response.json()["matches"] == []


async def test_unknown_id_is_404(client: httpx.AsyncClient) -> None:
    body = assert_problem(await client.get(f"/api/v1/posts/{uuid4()}/similar"), 404)
    assert body["title"] == "Paper not found"
    assert_problem(await client.get(f"/api/v1/papers/{uuid4()}"), 404)


@pytest.mark.parametrize("path", ["/api/v1/posts/not-a-uuid/similar", "/api/v1/posts/123/similar"])
async def test_malformed_id_is_422(client: httpx.AsyncClient, path: str) -> None:
    assert_problem(await client.get(path), 422)


@pytest.mark.parametrize("limit", ["0", "51", "-1", "ten"])
async def test_invalid_limit_is_422(client: httpx.AsyncClient, limit: str) -> None:
    paper_id = await submit(client, "some text")
    assert_problem(await client.get(f"/api/v1/posts/{paper_id}/similar?limit={limit}"), 422)


# --- Cross-cutting -------------------------------------------------------------------


@pytest.mark.usefixtures("indexed_corpus")
async def test_data_survives_an_application_restart(settings: Settings) -> None:
    async with running_app(settings) as first:
        paper_id = await submit(first, "galaxy cluster lensing and dark matter")
        before = (await first.get(f"/api/v1/posts/{paper_id}/similar")).json()

    async with running_app(settings) as second:  # new pools, clients and model instance
        after = (await second.get(f"/api/v1/posts/{paper_id}/similar")).json()

    assert after == before
    assert after["matches"][0]["arxiv_id"] == "2401.00003"


async def test_request_id_is_generated_or_propagated(client: httpx.AsyncClient) -> None:
    generated = await client.get("/api/v1/papers/" + str(uuid4()))
    assert len(generated.headers["x-request-id"]) == 32

    propagated = await client.get(
        "/api/v1/papers/" + str(uuid4()), headers={"X-Request-ID": "trace-abc.123"}
    )
    assert propagated.headers["x-request-id"] == "trace-abc.123"
    assert propagated.json()["request_id"] == "trace-abc.123"


async def test_unknown_route_is_problem_json(client: httpx.AsyncClient) -> None:
    assert_problem(await client.get("/api/v1/nope"), 404)


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
async def test_vector_store_down_gives_503(settings: Settings, resources: Resources) -> None:
    async with running_app(settings) as healthy:
        paper_id = await submit(healthy, "text stored while Qdrant was up")

    dead_qdrant = settings.model_copy(
        update={"qdrant_url": "http://127.0.0.1:9", "qdrant_timeout_seconds": 2}
    )
    async with running_app(dead_qdrant) as api:
        # Submitting needs only Postgres, so it still works...
        assert (
            await api.post("/api/v1/papers", json={"text": "still accepted"})
        ).status_code == 201
        # ...while searching needs Qdrant.
        response = await api.get(f"/api/v1/posts/{paper_id}/similar")
    body = assert_problem(response, 503)
    assert response.headers["retry-after"] == "5"
    assert "vector store" in str(body["detail"])


async def test_database_down_gives_503(settings: Settings, resources: Resources) -> None:
    dead_db = settings.model_copy(
        update={"database_url": "postgresql+asyncpg://pulse:pulse@127.0.0.1:9/pulse"}
    )
    async with running_app(dead_db) as api:
        response = await api.post("/api/v1/papers", json={"text": "no database"})
    body = assert_problem(response, 503)
    assert "database" in str(body["detail"])


# --- Operations ------------------------------------------------------------------------


async def test_health_is_always_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.usefixtures("indexed_corpus")
async def test_ready_reports_checks_and_index_stats(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"database": "ok", "vector_store": "ok"}
    assert body["index"] == {"papers": 4, "papers_indexed": 4, "vectors": 4}


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
async def test_ready_is_503_when_a_dependency_is_down(
    settings: Settings, resources: Resources
) -> None:
    dead_qdrant = settings.model_copy(
        update={"qdrant_url": "http://127.0.0.1:9", "qdrant_timeout_seconds": 1}
    )
    async with running_app(dead_qdrant) as api:
        response = await api.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["vector_store"].startswith("error")


async def test_metrics_are_exposed(client: httpx.AsyncClient) -> None:
    await submit(client, "count me")
    text = (await client.get("/metrics")).text
    assert 'http_requests_total{handler="/api/v1/papers",method="POST",status="201"}' in text
    assert 'pulse_embeddings_computed_total{purpose="submission_chunk"}' in text


@pytest.mark.usefixtures("indexed_corpus")
async def test_collection_wiped_while_running_is_recreated(
    client: httpx.AsyncClient, resources: Resources
) -> None:
    paper_id = await submit(client, "protein folding in yeast cells")
    await resources.qdrant.delete_collection(resources.settings.qdrant_collection)

    response = await client.get(f"/api/v1/posts/{paper_id}/similar")

    assert response.status_code == 200  # recreated and answered, not a 5xx loop
    assert response.json()["matches"] == []
    assert await resources.qdrant.collection_exists(resources.settings.qdrant_collection)


async def test_startup_refuses_an_index_built_by_another_model(
    settings: Settings, resources: Resources
) -> None:
    # Same dimension, different model: vectors would be silently incomparable.
    await resources.qdrant.delete_collection(settings.qdrant_collection)
    await resources.qdrant.create_collection(
        settings.qdrant_collection,
        vectors_config=VectorParams(size=HashingEmbedder.dimension, distance=Distance.COSINE),
        metadata={"embedding_model": "some/other-model"},
    )

    with pytest.raises(IndexModelMismatchError, match="some/other-model"):
        async with running_app(settings):
            pass


@pytest.mark.usefixtures("indexed_corpus")
async def test_submission_from_an_old_model_is_reembedded_once_and_saved(
    client: httpx.AsyncClient, resources: Resources
) -> None:
    paper_id = await submit(client, "protein folding in yeast cells")
    async with resources.session_factory() as session, session.begin():
        await session.execute(
            update(SubmissionRow)
            .where(SubmissionRow.id == paper_id)
            .values(embedding_model="old/model")
        )

    first = (await client.get(f"/api/v1/posts/{paper_id}/similar")).json()

    assert first["embedding_model"] == HashingEmbedder.model_name  # the model actually used
    assert first["matches"][0]["arxiv_id"] == "2401.00002"
    stored = (await client.get(f"/api/v1/papers/{paper_id}")).json()
    assert stored["embedding_model"] == HashingEmbedder.model_name  # saved, not redone per call
    assert stored["chunks"] == 1
    assert (await client.get(f"/api/v1/posts/{paper_id}/similar")).json() == first


async def test_unexpected_error_is_500_with_request_id_in_response_and_logs(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom(self: SubmissionService, submission_id: UUID) -> None:
        raise RuntimeError("bug: secret internal detail")

    monkeypatch.setattr(SubmissionService, "get", boom)

    response = await client.get(f"/api/v1/papers/{uuid4()}", headers={"X-Request-ID": "req-500"})

    body = assert_problem(response, 500)
    assert body["request_id"] == "req-500"
    assert "secret" not in response.text  # no internals leak to the client
    # The logged traceback carries the id the client was told to quote.
    [record] = [
        r for r in caplog.records if isinstance(r.msg, dict) and r.msg["event"] == "unhandled_error"
    ]
    event: dict[str, object] = record.msg  # type: ignore[assignment]
    assert event["request_id"] == "req-500"
    assert record.exc_info is not None


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("connect timed out"),
        PoolTimeoutError("QueuePool limit of size 10 overflow 5 reached"),
    ],
)
async def test_database_timeouts_are_503(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def slow_db(self: SubmissionService, submission_id: UUID) -> None:
        raise error

    monkeypatch.setattr(SubmissionService, "get", slow_db)

    response = await client.get(f"/api/v1/papers/{uuid4()}")

    assert_problem(response, 503)
    assert response.headers["retry-after"] == "5"

"""End-to-end ingestion against real Postgres and Qdrant; arXiv itself is simulated."""

from collections.abc import AsyncIterator

import httpx
import pyarrow.parquet as pq
import pytest
import respx
from sqlalchemy import select, update

from paper_similarity.bootstrap import Resources
from paper_similarity.db.models import ArxivPaperRow
from paper_similarity.db.repositories import PaperRepository
from paper_similarity.ingestion.arxiv_client import ArxivClient
from paper_similarity.ingestion.parquet_store import RawParquetStore
from paper_similarity.ingestion.pipeline import IngestionPipeline
from paper_similarity.services.indexing import IndexingService
from tests.conftest import FixtureLoader

ARXIV = "https://arxiv.test/api/query"


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def arxiv() -> respx.Router:
    """Simulated arXiv, wired only into the arXiv client's transport (Qdrant stays real)."""
    return respx.Router(assert_all_called=False)


@pytest.fixture
async def http(arxiv: respx.Router) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(arxiv.async_handler)) as client:
        yield client


def make_pipeline(
    resources: Resources, http: httpx.AsyncClient, page_size: int = 50
) -> IngestionPipeline:
    settings = resources.settings
    return IngestionPipeline(
        ArxivClient(
            http,
            base_url=ARXIV,
            page_size=page_size,
            max_attempts=2,
            backoff_base=0,
            sleep=no_sleep,
        ),
        resources.session_factory,
        RawParquetStore(settings.raw_data_dir),
        IndexingService(resources.session_factory, resources.embedder, resources.vector_store),
    )


async def db_counts(resources: Resources) -> tuple[int, int]:
    async with resources.session_factory() as session:
        return await PaperRepository(session).counts()


async def test_ingest_stores_lands_and_indexes(
    resources: Resources,
    http: httpx.AsyncClient,
    arxiv: respx.Router,
    fixture_bytes: FixtureLoader,
) -> None:
    arxiv.get(ARXIV).respond(200, content=fixture_bytes("arxiv_page.xml"))

    report = await make_pipeline(resources, http).run("cat:cs.CL", max_results=3)

    assert report.succeeded
    assert (report.fetched, report.inserted, report.indexed) == (3, 3, 3)
    assert await db_counts(resources) == (3, 3)
    assert await resources.vector_store.count() == 3

    [raw_file] = report.raw_files
    assert raw_file.parent.name.startswith("run_ts=")
    table = pq.read_table(raw_file)
    assert table.num_rows == 3
    assert set(table.column("query").to_pylist()) == {"cat:cs.CL"}


async def test_reingesting_the_same_papers_creates_no_duplicates(
    resources: Resources,
    http: httpx.AsyncClient,
    arxiv: respx.Router,
    fixture_bytes: FixtureLoader,
) -> None:
    arxiv.get(ARXIV).respond(200, content=fixture_bytes("arxiv_page.xml"))
    pipeline = make_pipeline(resources, http)

    await pipeline.run("cat:cs.CL", max_results=3)
    second = await pipeline.run("cat:cs.CL", max_results=3)

    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 3)
    assert second.indexed == 0  # nothing changed, nothing re-embedded
    assert await db_counts(resources) == (3, 3)
    assert await resources.vector_store.count() == 3


async def test_changed_abstract_is_reembedded(
    resources: Resources,
    http: httpx.AsyncClient,
    arxiv: respx.Router,
    fixture_bytes: FixtureLoader,
) -> None:
    original = fixture_bytes("arxiv_page.xml")
    revised = (
        original.replace(
            b"<id>http://arxiv.org/abs/2611.08433v1</id>",
            b"<id>http://arxiv.org/abs/2611.08433v2</id>",
        )
        .replace(b"LLM-based medical assistants", b"Revised: LLM-based medical assistants")
        .replace(
            b"<updated>2026-11-05T18:27:20Z</updated>",
            b"<updated>2026-11-06T09:00:00Z</updated>",
            1,
        )
    )
    arxiv.get(ARXIV).mock(
        side_effect=[httpx.Response(200, content=original), httpx.Response(200, content=revised)]
    )
    pipeline = make_pipeline(resources, http)

    await pipeline.run("cat:cs.CL", max_results=3)
    second = await pipeline.run("cat:cs.CL", max_results=3)

    assert (second.inserted, second.updated, second.unchanged) == (0, 1, 2)
    assert second.indexed == 1
    async with resources.session_factory() as session:
        row = await session.scalar(
            select(ArxivPaperRow).where(ArxivPaperRow.arxiv_id == "2611.08433")
        )
    assert row is not None
    assert row.version == 2
    assert row.abstract.startswith("Revised:")
    assert row.indexed_at is not None
    assert await resources.vector_store.count() == 3


async def test_arxiv_failure_midway_keeps_and_indexes_earlier_pages(
    resources: Resources,
    http: httpx.AsyncClient,
    arxiv: respx.Router,
    fixture_bytes: FixtureLoader,
) -> None:
    arxiv.get(ARXIV).mock(
        side_effect=[
            httpx.Response(200, content=fixture_bytes("arxiv_page.xml")),
            httpx.Response(503),
            httpx.Response(503),
        ]
    )

    report = await make_pipeline(resources, http, page_size=3).run("cat:cs.CL", max_results=6)

    assert not report.succeeded
    assert report.error is not None
    assert "after 2 attempts" in report.error
    assert (report.inserted, report.indexed) == (3, 3)
    assert await db_counts(resources) == (3, 3)


async def test_index_heals_rows_left_pending_by_an_interrupted_run(
    resources: Resources,
    http: httpx.AsyncClient,
    arxiv: respx.Router,
    fixture_bytes: FixtureLoader,
) -> None:
    arxiv.get(ARXIV).respond(200, content=fixture_bytes("arxiv_page.xml"))
    await make_pipeline(resources, http).run("cat:cs.CL", max_results=3)
    # Simulate a crash between the Qdrant upsert and marking rows as indexed.
    async with resources.session_factory() as session, session.begin():
        await session.execute(update(ArxivPaperRow).values(indexed_at=None))

    indexer = IndexingService(resources.session_factory, resources.embedder, resources.vector_store)
    assert await indexer.index_pending() == 3
    assert await db_counts(resources) == (3, 3)
    assert await resources.vector_store.count() == 3  # overwritten, not duplicated

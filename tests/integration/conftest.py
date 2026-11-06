"""Integration fixtures: real Postgres and Qdrant in throwaway Docker containers.

The containers start once per test session; every test starts from empty tables and an empty
collection. The schema is created by running the Alembic migrations, so the migrations are
tested too. Requires a running Docker daemon.
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.qdrant import QdrantContainer

from paper_similarity.bootstrap import Resources, create_resources
from paper_similarity.core.config import Settings
from tests.fakes import HashingEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_IMAGE = "qdrant/qdrant:v1.20.0"
POSTGRES_IMAGE = "postgres:16-alpine"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in str(item.path):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
        config.attributes["database_url"] = url
        config.attributes["configure_logger"] = False
        command.upgrade(config, "head")
        yield url


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    with QdrantContainer(QDRANT_IMAGE) as qdrant:
        yield f"http://{qdrant.get_container_host_ip()}:{qdrant.exposed_rest_port}"


@pytest.fixture
def settings(postgres_url: str, qdrant_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=postgres_url,
        qdrant_url=qdrant_url,
        qdrant_collection="test_papers",
        raw_data_dir=tmp_path / "raw",
        arxiv_base_url="https://arxiv.test/api/query",
        arxiv_request_delay_seconds=0,
        log_format="console",
        _env_file=None,
    )


@pytest.fixture
async def resources(settings: Settings) -> AsyncIterator[Resources]:
    res = create_resources(settings, embedder=HashingEmbedder())
    async with res.engine.begin() as conn:
        await conn.execute(text("TRUNCATE arxiv_papers, submissions, submission_chunks CASCADE"))
    await res.vector_store.recreate()
    yield res
    await res.close()

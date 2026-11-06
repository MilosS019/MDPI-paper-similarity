"""CLI wiring: output, exit codes and argument validation (no Docker, no network)."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paper_similarity import cli
from paper_similarity.bootstrap import Resources, create_resources
from paper_similarity.core.config import Settings, get_settings
from paper_similarity.domain.models import ArxivPaper
from paper_similarity.ingestion.parquet_store import RawParquetStore
from tests.fakes import HashingEmbedder

# The unreachable-Qdrant client warns about its version check, sometimes during a later test.
pytestmark = pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")

runner = CliRunner()
UNREACHABLE = "127.0.0.1:9"  # discard port: connections are refused immediately


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("PULSE_RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("PULSE_DATABASE_URL", f"postgresql+asyncpg://u:p@{UNREACHABLE}/db")
    monkeypatch.setenv("PULSE_QDRANT_URL", f"http://{UNREACHABLE}")
    monkeypatch.setenv("PULSE_LOG_FORMAT", "console")

    def fake_resources(settings: Settings) -> Resources:
        return create_resources(settings, embedder=HashingEmbedder())

    monkeypatch.setattr(cli, "create_resources", fake_resources)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_stats_without_data(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0
    assert "No raw data" in result.output


def test_stats_summarises_the_landing_zone(tmp_path: Path) -> None:
    when = datetime(2024, 1, 1, tzinfo=UTC)
    paper = ArxivPaper(
        arxiv_id="2401.00001",
        version=1,
        title="T",
        abstract="A",
        authors=(),
        categories=("cs.CL",),
        primary_category="cs.CL",
        published=when,
        updated=when,
        abs_url="https://arxiv.org/abs/2401.00001",
        pdf_url=None,
    )
    RawParquetStore(tmp_path / "raw").new_run("q").write_page([paper])

    result = runner.invoke(cli.app, ["stats"])

    assert result.exit_code == 0
    assert "runs=1 records=1 distinct_papers=1" in result.output
    assert "cs.CL" in result.output


def test_unreachable_services_give_a_clear_error_and_exit_code_1() -> None:
    result = runner.invoke(cli.app, ["index"])
    assert result.exit_code == 1
    assert "Indexing failed:" in result.output
    assert "Traceback" not in result.output


def test_invalid_arguments_are_rejected() -> None:
    assert runner.invoke(cli.app, ["ingest"]).exit_code == 2  # --query is required
    bad_sort = runner.invoke(cli.app, ["ingest", "-q", "cat:cs.CL", "--sort-by", "random"])
    assert bad_sort.exit_code == 2
    assert runner.invoke(cli.app, ["ingest", "-q", "cat:cs.CL", "--max", "0"]).exit_code == 2

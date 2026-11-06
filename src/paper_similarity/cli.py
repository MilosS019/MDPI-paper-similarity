"""Command line interface: ``pulse ingest | index | reindex | stats``.

Ingestion is a command rather than something the API does at startup: it is explicit,
repeatable and safe to re-run, does not slow down or block API startup, and maps directly to a
scheduled job (cron, Kubernetes CronJob) in production.
"""

import asyncio
from enum import StrEnum
from typing import Annotated

import httpx
import typer
from sqlalchemy.exc import SQLAlchemyError

from paper_similarity.bootstrap import Resources, create_resources
from paper_similarity.core.config import Settings, get_settings
from paper_similarity.core.logging import configure_logging
from paper_similarity.domain.errors import PaperSimilarityError
from paper_similarity.ingestion.arxiv_client import ArxivClient, SortBy
from paper_similarity.ingestion.parquet_store import RawParquetStore
from paper_similarity.ingestion.pipeline import IngestionPipeline, IngestionReport
from paper_similarity.ingestion.raw_reader import RawParquetReader
from paper_similarity.services.indexing import IndexingService
from paper_similarity.services.reindex import ReindexReport, ReindexService

app = typer.Typer(
    help="Scientific paper similarity service: ingestion and maintenance commands.",
    no_args_is_help=True,
    add_completion=False,
)


class SortOption(StrEnum):
    submitted = "submittedDate"
    updated = "lastUpdatedDate"
    relevance = "relevance"


# Expected operational failures: reported as one clear line and a non-zero exit code instead
# of a traceback. Anything else is a bug and keeps its traceback.
FAILURES = (PaperSimilarityError, SQLAlchemyError, OSError)

USER_AGENT = "paper-similarity/0.1 (arXiv ingestion for a similarity search demo)"


def _setup() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return settings


def _indexer(resources: Resources) -> IndexingService:
    return IndexingService(resources.session_factory, resources.embedder, resources.vector_store)


async def _ingest(
    settings: Settings, query: str, max_results: int, sort_by: SortBy
) -> IngestionReport:
    resources = create_resources(settings)
    try:
        await resources.vector_store.ensure_collection()
        async with httpx.AsyncClient(
            timeout=settings.arxiv_timeout_seconds,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as http:
            client = ArxivClient(
                http,
                base_url=settings.arxiv_base_url,
                page_size=settings.arxiv_page_size,
                request_delay=settings.arxiv_request_delay_seconds,
                max_attempts=settings.arxiv_max_retries,
            )
            pipeline = IngestionPipeline(
                client,
                resources.session_factory,
                RawParquetStore(settings.raw_data_dir),
                _indexer(resources),
            )
            return await pipeline.run(query, max_results, sort_by=sort_by)
    finally:
        await resources.close()


@app.command()
def ingest(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help='arXiv search query, e.g. "cat:cs.CL AND abs:retrieval". '
            "Syntax: https://info.arxiv.org/help/api/user-manual.html#query_details",
        ),
    ],
    max_results: Annotated[
        int, typer.Option("--max", "-n", min=1, max=10_000, help="Papers to fetch.")
    ] = 50,
    sort_by: Annotated[
        SortOption, typer.Option(help="Result order; dates are newest first.")
    ] = SortOption.submitted,
) -> None:
    """Fetch papers from arXiv, store them, and index their embeddings. Safe to re-run."""
    settings = _setup()
    try:
        report = asyncio.run(_ingest(settings, query, max_results, sort_by.value))
    except FAILURES as exc:
        typer.secho(f"Ingestion failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"fetched={report.fetched} inserted={report.inserted} updated={report.updated} "
        f"unchanged={report.unchanged} skipped={report.skipped} indexed={report.indexed} "
        f"raw_files={len(report.raw_files)}"
    )
    if not report.succeeded:
        typer.secho(
            f"Ingestion incomplete: {report.error}. Papers fetched before the failure were "
            "saved and indexed; re-run the command to continue.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)


async def _index(settings: Settings) -> int:
    resources = create_resources(settings)
    try:
        await resources.vector_store.ensure_collection()
        return await _indexer(resources).index_pending()
    finally:
        await resources.close()


@app.command()
def index() -> None:
    """Index papers that are stored but not yet (or no longer) in the vector store."""
    settings = _setup()
    try:
        count = asyncio.run(_index(settings))
    except FAILURES as exc:
        typer.secho(f"Indexing failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"indexed={count}")


async def _reindex(settings: Settings, from_raw: bool) -> ReindexReport:
    resources = create_resources(settings)
    try:
        service = ReindexService(
            resources.session_factory,
            RawParquetReader(settings.raw_data_dir),
            _indexer(resources),
            resources.vector_store,
        )
        return await service.run(from_raw=from_raw)
    finally:
        await resources.close()


@app.command()
def reindex(
    from_raw: Annotated[
        bool,
        typer.Option(
            "--from-raw/--from-db",
            help="Reload papers from the raw Parquet files (via DuckDB) before re-embedding, "
            "or re-embed what is already in Postgres.",
        ),
    ] = True,
) -> None:
    """Rebuild the vector index (e.g. after changing the embedding model) without calling arXiv.

    Drops and recreates the Qdrant collection, then re-embeds every paper.
    """
    settings = _setup()
    try:
        report = asyncio.run(_reindex(settings, from_raw))
    except FAILURES as exc:
        typer.secho(f"Reindex failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"raw_papers={report.raw_papers} inserted={report.inserted} updated={report.updated} "
        f"unchanged={report.unchanged} indexed={report.indexed}"
    )


@app.command()
def stats() -> None:
    """Summarise the raw landing zone with DuckDB: runs, papers, top categories."""
    settings = _setup()
    summary = RawParquetReader(settings.raw_data_dir).summary()
    if summary is None:
        typer.echo(f"No raw data under {settings.raw_data_dir}. Run `pulse ingest` first.")
        return
    typer.echo(f"runs={summary.runs} records={summary.records} distinct_papers={summary.papers}")
    typer.echo(f"published: {summary.oldest} .. {summary.newest}")
    typer.echo("top primary categories:")
    for category, count in summary.top_categories:
        typer.echo(f"  {category or '-':<12} {count}")


if __name__ == "__main__":  # pragma: no cover
    app()

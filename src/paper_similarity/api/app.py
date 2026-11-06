"""Application factory and lifespan."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from paper_similarity.api.errors import register_error_handlers
from paper_similarity.api.middleware import RequestContextMiddleware
from paper_similarity.api.routes_papers import router as papers_router
from paper_similarity.bootstrap import Resources, create_resources
from paper_similarity.core.config import Settings, get_settings
from paper_similarity.core.logging import configure_logging
from paper_similarity.domain.errors import DependencyUnavailableError

log = structlog.get_logger(__name__)

ResourcesFactory = Callable[[Settings], Resources]


def create_app(
    settings: Settings | None = None,
    resources_factory: ResourcesFactory | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own settings and a factory that injects a fake embedder."""
    settings = settings or get_settings()
    factory = resources_factory or create_resources
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load the model and open the pools once per process, not per request.
        resources = factory(settings)
        app.state.resources = resources
        try:
            await resources.vector_store.ensure_collection()
        except DependencyUnavailableError as exc:
            # Start anyway: /ready reports the problem and requests get a clear 503 until
            # Qdrant is reachable, instead of the process crash-looping.
            log.warning("vector_store_unavailable_at_startup", error=str(exc))
        log.info("app_started", embedding_model=resources.embedder.model_name)
        try:
            yield
        finally:
            await resources.close()
            log.info("app_stopped")

    app = FastAPI(
        title="Paper Similarity Service",
        version="0.1.0",
        summary="Find arXiv papers similar to a submitted manuscript.",
        description=(
            "Submit a paper with `POST /api/v1/papers`, then fetch ranked similar arXiv "
            "papers with `GET /api/v1/posts/{id}/similar`. Errors use RFC 9457 "
            "`application/problem+json`."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(papers_router)
    return app

"""Liveness and readiness probes (Kubernetes-style).

* ``/health``: the process is up and serving HTTP. No dependency checks, so a slow database
  never makes the orchestrator restart a healthy process.
* ``/ready``: the service can do useful work (Postgres and Qdrant reachable). A failing
  readiness check takes the instance out of load balancing without restarting it.
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from paper_similarity.api.deps import ResourcesDep
from paper_similarity.db.repositories import PaperRepository

router = APIRouter(tags=["operations"])

CHECK_TIMEOUT_SECONDS = 3.0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class IndexStats(BaseModel):
    papers: int
    papers_indexed: int
    vectors: int


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    checks: dict[str, str]
    embedding_model: str
    index: IndexStats | None = None


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is unavailable"}},
    summary="Readiness probe with index statistics",
)
async def ready(resources: ResourcesDep, response: Response) -> ReadinessResponse:
    checks: dict[str, str] = {}
    papers = indexed = vectors = None

    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            async with resources.session_factory() as session:
                await session.execute(text("SELECT 1"))
                papers, indexed = await PaperRepository(session).counts()
        checks["database"] = "ok"
    except Exception as exc:  # a probe reports failures, it never raises
        checks["database"] = f"error: {type(exc).__name__}"

    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            await resources.vector_store.ping()
            vectors = await resources.vector_store.count()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {type(exc).__name__}"

    is_ready = all(v == "ok" for v in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    stats = (
        IndexStats(papers=papers, papers_indexed=indexed, vectors=vectors)
        if papers is not None and indexed is not None and vectors is not None
        else None
    )
    return ReadinessResponse(
        status="ready" if is_ready else "unavailable",
        checks=checks,
        embedding_model=resources.embedder.model_name,
        index=stats,
    )

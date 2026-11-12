"""Liveness and readiness probes (Kubernetes-style).

* ``/health``: the process is up and serving HTTP. No dependency checks, so a slow database
  never makes the orchestrator restart a healthy process.
* ``/ready``: the service can do useful work (Postgres and Qdrant reachable). A failing
  readiness check takes the instance out of load balancing without restarting it.
* ``/stats``: how many papers are stored and indexed. Kept out of ``/ready`` on purpose: a
  probe runs every few seconds on every replica, and exact counts scan the whole table and
  collection.
"""

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from paper_similarity.api.deps import ResourcesDep
from paper_similarity.api.schemas import Problem
from paper_similarity.db.repositories import PaperRepository

router = APIRouter(tags=["operations"])

CHECK_TIMEOUT_SECONDS = 3.0


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    checks: dict[str, str]
    embedding_model: str


class IndexStats(BaseModel):
    embedding_model: str
    papers: int
    papers_indexed: int
    vectors: int


_UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "Database or vector store unavailable"}
}


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is unavailable"}},
    summary="Readiness probe",
)
async def ready(resources: ResourcesDep, response: Response) -> ReadinessResponse:
    checks: dict[str, str] = {}

    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            async with resources.session_factory() as session:
                await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # a probe reports failures, it never raises
        checks["database"] = f"error: {type(exc).__name__}"

    try:
        async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
            await resources.vector_store.ping()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {type(exc).__name__}"

    is_ready = all(v == "ok" for v in checks.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if is_ready else "unavailable",
        checks=checks,
        embedding_model=resources.embedder.model_name,
    )


@router.get(
    "/stats",
    response_model=IndexStats,
    responses=_UNAVAILABLE,
    summary="Index statistics",
)
async def stats(resources: ResourcesDep) -> IndexStats:
    """Papers stored, papers whose vector is current, and vectors in the index. The three
    agree when the index is in sync; ``papers_indexed`` above ``vectors`` means the collection
    lost points and needs ``pulse reindex --from-db``."""
    model = resources.embedder.model_name
    async with resources.session_factory() as session:
        papers, indexed = await PaperRepository(session).counts(model)
    vectors = await resources.vector_store.count()
    return IndexStats(embedding_model=model, papers=papers, papers_indexed=indexed, vectors=vectors)

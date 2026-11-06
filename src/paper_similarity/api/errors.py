"""Map exceptions to RFC 9457 ``application/problem+json`` responses.

Every error has the same shape, carries the request id for log correlation, and never leaks
internal details (stack traces, SQL) to the client: those go to the logs.
"""

import socket
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from paper_similarity.api.schemas import Problem
from paper_similarity.domain.errors import (
    DependencyUnavailableError,
    EmptyTextError,
    IndexModelMismatchError,
    SubmissionNotFoundError,
    VectorStoreError,
)

log = structlog.get_logger(__name__)

PROBLEM_JSON = "application/problem+json"
PROBLEM_BASE = "https://paper-similarity.example/problems/"


class TextTooLongError(Exception):
    def __init__(self, length: int, limit: int) -> None:
        super().__init__(f"text has {length} characters; the maximum is {limit}")


def problem_response(
    request: Request,
    status: int,
    title: str,
    detail: str | None = None,
    *,
    problem_type: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = Problem(
        type=f"{PROBLEM_BASE}{problem_type}" if problem_type else "about:blank",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        request_id=getattr(request.state, "request_id", None),
        errors=errors,
    )
    return JSONResponse(
        jsonable_encoder(body, exclude_none=True),
        status_code=status,
        media_type=PROBLEM_JSON,
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return problem_response(
            request,
            422,
            "Invalid request",
            "The request did not pass validation; see 'errors'.",
            problem_type="validation-error",
            errors=errors,
        )

    @app.exception_handler(TextTooLongError)
    async def _too_long(request: Request, exc: TextTooLongError) -> JSONResponse:
        return problem_response(
            request, 422, "Text too long", str(exc), problem_type="text-too-long"
        )

    @app.exception_handler(EmptyTextError)
    async def _empty(request: Request, exc: EmptyTextError) -> JSONResponse:
        return problem_response(request, 422, "Empty text", str(exc), problem_type="empty-text")

    @app.exception_handler(SubmissionNotFoundError)
    async def _not_found(request: Request, exc: SubmissionNotFoundError) -> JSONResponse:
        return problem_response(
            request, 404, "Paper not found", str(exc), problem_type="paper-not-found"
        )

    @app.exception_handler(DependencyUnavailableError)
    async def _dependency(request: Request, exc: DependencyUnavailableError) -> JSONResponse:
        log.error("dependency_unavailable", dependency=exc.dependency, error=exc.detail)
        return problem_response(
            request,
            503,
            "Service temporarily unavailable",
            f"The {exc.dependency} is not reachable. Please retry shortly.",
            problem_type="dependency-unavailable",
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(IndexModelMismatchError)
    async def _index_mismatch(request: Request, exc: IndexModelMismatchError) -> JSONResponse:
        # A deployment problem, not an outage: no Retry-After, operators must reindex.
        log.error("search_index_model_mismatch", error=str(exc))
        return problem_response(
            request,
            503,
            "Search index unavailable",
            "The search index does not match the configured embedding model.",
            problem_type="index-mismatch",
        )

    @app.exception_handler(VectorStoreError)
    async def _vector_store(request: Request, exc: VectorStoreError) -> JSONResponse:
        log.error("vector_store_error", error=str(exc))
        return problem_response(
            request, 500, "Internal server error", "The search request could not be completed."
        )

    # SQLAlchemy wraps most driver errors, but asyncpg raises plain socket errors when it
    # cannot connect at all. Qdrant failures are already translated in the vector store, so
    # an untranslated connection error here comes from the database.
    @app.exception_handler(OperationalError)
    @app.exception_handler(InterfaceError)
    @app.exception_handler(ConnectionError)
    @app.exception_handler(socket.gaierror)
    async def _database(request: Request, exc: Exception) -> JSONResponse:
        cause = exc.orig if isinstance(exc, DBAPIError) else exc
        log.error("database_unavailable", error=f"{type(cause).__name__}: {cause}")
        return problem_response(
            request,
            503,
            "Service temporarily unavailable",
            "The database is not reachable. Please retry shortly.",
            problem_type="dependency-unavailable",
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem_response(
            request, exc.status_code, str(exc.detail), headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", error_type=type(exc).__name__)
        return problem_response(
            request,
            500,
            "Internal server error",
            "An unexpected error occurred. Quote the request id when reporting it.",
        )

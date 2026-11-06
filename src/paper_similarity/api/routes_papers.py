"""Paper submission and similarity endpoints."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Path, Query, Response, status

from paper_similarity.api.deps import ResourcesDep, SubmissionServiceDep
from paper_similarity.api.errors import TextTooLongError
from paper_similarity.api.schemas import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    Match,
    Problem,
    SimilarPapersResponse,
    SubmissionResponse,
    SubmitPaperRequest,
    SubmitPaperResponse,
)

router = APIRouter(prefix="/api/v1", tags=["papers"])

PaperId = Annotated[UUID, Path(description="Identifier returned by POST /api/v1/papers.")]
Limit = Annotated[
    int, Query(ge=1, le=MAX_LIMIT, description="Maximum number of matches to return.")
]

_ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": Problem, "description": "Invalid request"},
    503: {"model": Problem, "description": "Database or vector store unavailable"},
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": Problem, "description": "Unknown paper id"}
}


@router.post(
    "/papers",
    status_code=status.HTTP_201_CREATED,
    response_model=SubmitPaperResponse,
    responses=_ERRORS,
    summary="Submit a paper or manuscript",
)
async def submit_paper(
    body: SubmitPaperRequest,
    service: SubmissionServiceDep,
    resources: ResourcesDep,
    response: Response,
) -> SubmitPaperResponse:
    """Validate, embed and store a manuscript. Returns the id to use with the similarity
    endpoint."""
    limit = resources.settings.max_submission_chars
    if len(body.text) > limit:
        raise TextTooLongError(len(body.text), limit)

    submission = await service.create(body.title, body.text)
    response.headers["Location"] = f"/api/v1/papers/{submission.id}"
    return SubmitPaperResponse(
        id=submission.id,
        chunks=len(submission.chunks),
        embedding_model=submission.embedding_model,
        created_at=submission.created_at,
    )


@router.get(
    "/papers/{paper_id}",
    response_model=SubmissionResponse,
    responses={**_ERRORS, **_NOT_FOUND},
    summary="Get a submitted paper",
)
async def get_paper(paper_id: PaperId, service: SubmissionServiceDep) -> SubmissionResponse:
    submission = await service.get(paper_id)
    return SubmissionResponse(
        id=submission.id,
        title=submission.title,
        text_length=len(submission.text),
        chunks=len(submission.chunks),
        embedding_model=submission.embedding_model,
        created_at=submission.created_at,
    )


async def _similar(
    paper_id: UUID, limit: int, service: SubmissionServiceDep
) -> SimilarPapersResponse:
    submission, matches = await service.find_similar(paper_id, limit)
    return SimilarPapersResponse(
        paper_id=submission.id,
        embedding_model=submission.embedding_model,
        limit=limit,
        matches=[
            Match(
                arxiv_id=m.arxiv_id,
                title=m.title,
                authors=list(m.authors),
                url=m.url,
                published=m.published,
                primary_category=m.primary_category,
                score=round(m.score, 6),
                matched_chunk=m.matched_chunk,
            )
            for m in matches
        ],
    )


@router.get(
    "/posts/{paper_id}/similar",
    response_model=SimilarPapersResponse,
    responses={**_ERRORS, **_NOT_FOUND},
    summary="Find indexed arXiv papers similar to a submitted paper",
)
async def similar_papers(
    paper_id: PaperId, service: SubmissionServiceDep, limit: Limit = DEFAULT_LIMIT
) -> SimilarPapersResponse:
    """Ranked by cosine similarity, highest first (ties broken by arXiv id). For long
    submissions each paper is scored by its best-matching chunk.

    The path uses ``/posts`` as written in the assignment; ``/papers/{id}/similar`` is an
    equivalent alias that matches the submission endpoint's naming.
    """
    return await _similar(paper_id, limit, service)


@router.get(
    "/papers/{paper_id}/similar",
    response_model=SimilarPapersResponse,
    responses={**_ERRORS, **_NOT_FOUND},
    summary="Alias of /posts/{paper_id}/similar",
)
async def similar_papers_alias(
    paper_id: PaperId, service: SubmissionServiceDep, limit: Limit = DEFAULT_LIMIT
) -> SimilarPapersResponse:
    return await _similar(paper_id, limit, service)

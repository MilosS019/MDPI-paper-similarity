"""FastAPI dependencies: resolve long-lived resources and build per-request services."""

from typing import Annotated

from fastapi import Depends, Request

from paper_similarity.bootstrap import Resources
from paper_similarity.services.submissions import SubmissionService


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


def get_submission_service(resources: ResourcesDep) -> SubmissionService:
    settings = resources.settings
    return SubmissionService(
        resources.session_factory,
        resources.embedder,
        resources.vector_store,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


SubmissionServiceDep = Annotated[SubmissionService, Depends(get_submission_service)]

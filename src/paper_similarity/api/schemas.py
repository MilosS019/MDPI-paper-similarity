"""Request and response models: the public API contract, documented in the OpenAPI schema."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_LIMIT = 10
MAX_LIMIT = 50


class SubmitPaperRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "title": "Retrieval-augmented generation for scientific question answering",
                    "text": "We study how retrieving passages from scientific papers improves "
                    "the factual accuracy of large language models answering domain questions.",
                }
            ]
        },
    )

    title: str | None = Field(
        default=None, max_length=1000, description="Optional manuscript title."
    )
    text: str = Field(
        min_length=1,
        description="Abstract or full manuscript text. Long texts are split into chunks and "
        "every chunk is compared with the index. The maximum length is configurable "
        "(PULSE_MAX_SUBMISSION_CHARS, 100,000 characters by default).",
    )

    @field_validator("title", "text")
    @classmethod
    def _no_nul_characters(cls, value: str | None) -> str | None:
        # Valid JSON ("\u0000") but not storable in a Postgres text column: without this
        # check the insert fails and the client gets a 500 for its own bad input.
        if value is not None and "\x00" in value:
            raise ValueError("must not contain NUL (\\u0000) characters")
        return value

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("title")
    @classmethod
    def _blank_title_is_none(cls, value: str | None) -> str | None:
        return value if value and value.strip() else None


class SubmitPaperResponse(BaseModel):
    id: UUID = Field(description="Stable identifier, used with the similarity endpoint.")
    status: Literal["created"] = "created"
    chunks: int = Field(description="Number of chunks the text was split into for embedding.")
    embedding_model: str
    created_at: datetime


class SubmissionResponse(BaseModel):
    id: UUID
    title: str | None
    text_length: int
    chunks: int
    embedding_model: str
    created_at: datetime


class Match(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    url: str
    published: datetime
    primary_category: str | None
    score: float = Field(
        description="Cosine similarity in [-1, 1] between the best-matching chunk of the "
        "submission and the paper's title + abstract. Higher is more similar. Use it to rank, "
        "not as a calibrated probability."
    )
    matched_chunk: int = Field(description="Index of the submission chunk that matched best.")


class SimilarPapersResponse(BaseModel):
    paper_id: UUID
    embedding_model: str
    limit: int
    matches: list[Match]


class Problem(BaseModel):
    """RFC 9457 problem details, used for every error response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, object]] | None = None

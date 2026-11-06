"""Domain-level exceptions. The API layer maps them to HTTP responses."""


class PaperSimilarityError(Exception):
    """Base class for errors raised by this application."""


class SubmissionNotFoundError(PaperSimilarityError):
    def __init__(self, submission_id: object) -> None:
        super().__init__(f"Submission {submission_id} was not found")
        self.submission_id = submission_id


class EmptyTextError(PaperSimilarityError):
    """The text has no content left once whitespace is removed."""


class DependencyUnavailableError(PaperSimilarityError):
    """A backing service (database or vector store) could not be reached."""

    def __init__(self, dependency: str, detail: str) -> None:
        super().__init__(f"{dependency} is unavailable: {detail}")
        self.dependency = dependency
        self.detail = detail


class ArxivUnavailableError(PaperSimilarityError):
    """The arXiv API kept failing after all retries."""


class VectorStoreError(PaperSimilarityError):
    """The vector store rejected a request (a permanent error, not an outage)."""


class CollectionMissingError(VectorStoreError):
    """The vector collection does not exist (e.g. Qdrant storage was wiped)."""


class IndexModelMismatchError(VectorStoreError):
    """The vector index was built by a different embedding model; it must be rebuilt."""

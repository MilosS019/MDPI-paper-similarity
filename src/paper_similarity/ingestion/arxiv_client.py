"""HTTP client for the public arXiv API (https://info.arxiv.org/help/api/).

Reliability behaviour:

* Timeouts on every request.
* Retries with exponential backoff for transient failures: network errors, timeouts, HTTP 429
  and 5xx, arXiv's in-band error entries on a 2xx response, and arXiv's known glitch of
  returning an empty page although more results exist.
* A ``Retry-After`` header on a 429/5xx response is honoured: the wait is the longer of that
  and the backoff, capped so a bad header cannot stall a run.
* Client errors (HTTP 4xx, e.g. an invalid query) are not retried: retrying cannot fix them.
* At least ``request_delay`` seconds between requests, as arXiv's terms of use ask.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from paper_similarity.core import metrics
from paper_similarity.domain.errors import ArxivUnavailableError, PaperSimilarityError
from paper_similarity.domain.models import ArxivPaper
from paper_similarity.ingestion.parser import MalformedEntryError, parse_feed

log = structlog.get_logger(__name__)

SortBy = Literal["relevance", "lastUpdatedDate", "submittedDate"]
SortOrder = Literal["ascending", "descending"]
Sleep = Callable[[float], Awaitable[None]]


class ArxivQueryError(PaperSimilarityError):
    """arXiv rejected the request (e.g. invalid query syntax). Not retried."""


# Upper bound for any single wait between attempts, whatever the server asks for.
MAX_WAIT_SECONDS = 60.0


class _TransientArxivError(Exception):
    """Internal marker for failures worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        # Seconds the server asked us to wait (its Retry-After header), if it said.
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The ``Retry-After`` header as seconds. The HTTP-date form is ignored (backoff applies)."""
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if value.isdigit() else None


@dataclass(frozen=True, slots=True)
class ArxivPage:
    start: int
    total_results: int
    papers: list[ArxivPaper]
    skipped: list[str]


class ArxivClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        page_size: int = 50,
        request_delay: float = 3.0,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._base_url = base_url
        self._page_size = page_size
        self._request_delay = request_delay
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._last_request_at: float | None = None

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        sort_by: SortBy = "submittedDate",
        sort_order: SortOrder = "descending",
    ) -> AsyncIterator[ArxivPage]:
        """Yield result pages until ``max_results`` papers were returned or results run out."""
        start = 0
        returned = 0
        while returned < max_results:
            size = min(self._page_size, max_results - returned)
            page = await self._fetch_page(query, start, size, sort_by, sort_order)
            if not page.papers and not page.skipped:
                break  # no more results
            yield page
            fetched = len(page.papers) + len(page.skipped)
            returned += len(page.papers)
            start += fetched
            if start >= page.total_results:
                break

    async def _fetch_page(
        self, query: str, start: int, size: int, sort_by: SortBy, sort_order: SortOrder
    ) -> ArxivPage:
        params: dict[str, str | int] = {
            "search_query": query,
            "start": start,
            "max_results": size,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_TransientArxivError),
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            sleep=self._sleep,
            before_sleep=self._log_retry,
            reraise=False,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._request_page(params, start)
        except RetryError as exc:
            metrics.ARXIV_FAILURES.inc()
            cause = exc.last_attempt.exception()
            raise ArxivUnavailableError(
                f"arXiv request failed after {self._max_attempts} attempts: {cause}"
            ) from cause
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request_page(self, params: dict[str, str | int], start: int) -> ArxivPage:
        await self._respect_rate_limit()
        try:
            response = await self._http.get(self._base_url, params=params)
        except httpx.TransportError as exc:  # connect errors, timeouts, protocol errors
            raise _TransientArxivError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise _TransientArxivError(
                f"HTTP {response.status_code}", retry_after=_retry_after_seconds(response)
            )

        try:
            feed = parse_feed(response.content)
        except MalformedEntryError as exc:
            if response.is_success:
                raise _TransientArxivError(str(exc)) from exc
            feed = None

        if response.is_client_error:
            detail = feed.error_message if feed and feed.error_message else response.text[:200]
            raise ArxivQueryError(
                f"arXiv rejected the request (HTTP {response.status_code}): {detail}"
            )
        if feed is None:  # pragma: no cover - non-2xx/4xx/5xx statuses
            raise _TransientArxivError(f"unexpected HTTP {response.status_code}")
        if feed.error_message:
            raise _TransientArxivError(f"arXiv error: {feed.error_message}")
        if not feed.papers and not feed.skipped and start < feed.total_results:
            # Known arXiv behaviour: occasionally an empty page although results exist.
            raise _TransientArxivError("empty page before the end of the results")

        for reason in feed.skipped:
            log.warning("arxiv_entry_skipped", reason=reason)
        return ArxivPage(
            start=start,
            total_results=feed.total_results,
            papers=feed.papers,
            skipped=feed.skipped,
        )

    async def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self._request_delay - (now - self._last_request_at)
            if remaining > 0:
                await self._sleep(remaining)
        self._last_request_at = time.monotonic()

    def _wait(self, state: RetryCallState) -> float:
        """Exponential backoff, or the server's Retry-After if that is longer."""
        backoff = float(
            wait_exponential(multiplier=self._backoff_base, max=MAX_WAIT_SECONDS)(state)
        )
        exc = state.outcome.exception() if state.outcome else None
        requested = getattr(exc, "retry_after", None) or 0.0
        return min(max(backoff, requested), MAX_WAIT_SECONDS)

    @staticmethod
    def _log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "arxiv_request_retry",
            attempt=state.attempt_number,
            error=str(exc),
            wait_seconds=round(state.next_action.sleep, 2) if state.next_action else None,
        )

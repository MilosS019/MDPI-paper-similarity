"""arXiv client behaviour against simulated responses (respx). No network, no real sleeping."""

import httpx
import pytest
import respx

from paper_similarity.domain.errors import ArxivUnavailableError
from paper_similarity.ingestion.arxiv_client import ArxivClient, ArxivPage, ArxivQueryError
from tests.conftest import FixtureLoader

BASE_URL = "https://arxiv.test/api/query"


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_client(http: httpx.AsyncClient, sleep: RecordingSleep, **kwargs: object) -> ArxivClient:
    params: dict[str, object] = {
        "base_url": BASE_URL,
        "page_size": 3,
        "request_delay": 3.0,
        "max_attempts": 3,
        "backoff_base": 1.0,
        "sleep": sleep,
    }
    params.update(kwargs)
    return ArxivClient(http, **params)  # type: ignore[arg-type]


async def collect(client: ArxivClient, query: str = "cat:cs.CL", n: int = 3) -> list[ArxivPage]:
    return [page async for page in client.search(query, max_results=n)]


@respx.mock
async def test_fetches_and_parses_a_page(fixture_bytes: FixtureLoader) -> None:
    route = respx.get(BASE_URL).respond(200, content=fixture_bytes("arxiv_page.xml"))
    async with httpx.AsyncClient() as http:
        pages = await collect(make_client(http, RecordingSleep()))

    assert len(pages) == 1
    assert len(pages[0].papers) == 3
    request = route.calls.last.request
    assert request.url.params["search_query"] == "cat:cs.CL"
    assert request.url.params["max_results"] == "3"
    assert request.url.params["sortBy"] == "submittedDate"


@respx.mock
async def test_retries_transient_server_errors_then_succeeds(fixture_bytes: FixtureLoader) -> None:
    route = respx.get(BASE_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectTimeout("timed out"),
            httpx.Response(200, content=fixture_bytes("arxiv_page.xml")),
        ]
    )
    sleep = RecordingSleep()
    async with httpx.AsyncClient() as http:
        pages = await collect(make_client(http, sleep))

    assert route.call_count == 3
    assert len(pages[0].papers) == 3
    assert len(sleep.calls) >= 2  # backed off between attempts


@respx.mock
async def test_gives_up_after_max_attempts() -> None:
    route = respx.get(BASE_URL).respond(503)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ArxivUnavailableError, match="after 3 attempts"):
            await collect(make_client(http, RecordingSleep()))
    assert route.call_count == 3


@respx.mock
async def test_invalid_query_is_not_retried(fixture_bytes: FixtureLoader) -> None:
    route = respx.get(BASE_URL).respond(400, content=fixture_bytes("arxiv_error_invalid_query.xml"))
    async with httpx.AsyncClient() as http:
        with pytest.raises(ArxivQueryError, match="Invalid query string"):
            await collect(make_client(http, RecordingSleep()))
    assert route.call_count == 1


@respx.mock
async def test_in_band_error_on_200_is_retried(fixture_bytes: FixtureLoader) -> None:
    error_feed = fixture_bytes("arxiv_error_invalid_query.xml")
    route = respx.get(BASE_URL).mock(
        side_effect=[
            httpx.Response(200, content=error_feed),
            httpx.Response(200, content=fixture_bytes("arxiv_page.xml")),
        ]
    )
    async with httpx.AsyncClient() as http:
        pages = await collect(make_client(http, RecordingSleep()))
    assert route.call_count == 2
    assert len(pages[0].papers) == 3


@respx.mock
async def test_empty_result_set_ends_iteration(fixture_bytes: FixtureLoader) -> None:
    respx.get(BASE_URL).respond(200, content=fixture_bytes("arxiv_empty.xml"))
    async with httpx.AsyncClient() as http:
        pages = await collect(make_client(http, RecordingSleep()))
    assert pages == []


@respx.mock
async def test_paginates_and_waits_between_requests(fixture_bytes: FixtureLoader) -> None:
    page = fixture_bytes("arxiv_page.xml")
    route = respx.get(BASE_URL).respond(200, content=page)
    sleep = RecordingSleep()
    async with httpx.AsyncClient() as http:
        pages = await collect(make_client(http, sleep, page_size=3), n=6)

    assert route.call_count == 2
    starts = [call.request.url.params["start"] for call in route.calls]
    assert starts == ["0", "3"]
    assert sum(len(p.papers) for p in pages) == 6
    # The second request waited for (almost) the full polite delay.
    assert sleep.calls
    assert sleep.calls[0] == pytest.approx(3.0, abs=0.5)

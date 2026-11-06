"""Request id + access log middleware (pure ASGI, so it also wraps error responses)."""

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from paper_similarity.api.errors import PROBLEM_JSON, internal_error_body, problem_body

log = structlog.get_logger("paper_similarity.access")

REQUEST_ID_HEADER = "X-Request-ID"
# Accept a caller's id only if it looks harmless (it ends up in logs and response headers).
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    """Assigns every request an id, binds it to all log lines, and logs one access line.

    The id is taken from an incoming ``X-Request-ID`` header (e.g. set by a gateway) when it
    looks valid, otherwise generated, and is returned in the response header.

    It is also the last line of defence for unexpected exceptions: they are logged with the
    request id and answered with a problem+json 500 carrying the same id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers", [])).get(REQUEST_ID_HEADER.lower().encode(), b"")
        candidate = incoming.decode("latin-1")
        request_id = candidate if _VALID_ID.match(candidate) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            log.exception("unhandled_error", error_type=type(exc).__name__)
            if response_started:  # too late to send an error response; drop the connection
                raise
            body = internal_error_body(scope.get("path", ""), request_id)
            await send_wrapper(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", PROBLEM_JSON.encode()),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send_wrapper({"type": "http.response.body", "body": body})
        finally:
            path = scope.get("path", "")
            if path not in ("/health", "/metrics"):  # keep probe/scrape noise out of logs
                log.info(
                    "request",
                    method=scope.get("method"),
                    path=path,
                    status=status_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            structlog.contextvars.clear_contextvars()


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes`` with 413, before they are read and parsed.

    FastAPI parses the whole JSON body before a route can check the text length, so without
    this a client could make the server buffer and parse an arbitrarily large payload. A
    declared ``Content-Length`` is checked up front; chunked bodies are counted as they arrive
    (FastAPI passes an ``HTTPException`` raised while reading the body through to the handler).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail="Request body too large")
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        body = problem_body(
            status=413,
            title="Request body too large",
            detail=f"The request body exceeds {self.max_bytes} bytes.",
            path=scope.get("path", ""),
            request_id=scope.get("state", {}).get("request_id"),
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", PROBLEM_JSON.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

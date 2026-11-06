"""Request id + access log middleware (pure ASGI, so it also wraps error responses)."""

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from paper_similarity.api.errors import PROBLEM_JSON, internal_error_body

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

"""Structured logging configuration for the serving API.

Uses structlog for JSON-formatted log output with request ID propagation.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# Context variable for request ID propagation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def configure_logging() -> None:
    """Configure structlog for JSON output with request context."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a request ID to each request.

    - Reads X-Request-ID from incoming headers if present
    - Generates a UUID if not present
    - Binds request_id to structlog contextvars for all downstream logs
    - Returns request_id in X-Request-ID response header
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request_id_var.set(rid)
        structlog.contextvars.bind_contextvars(request_id=rid)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            structlog.contextvars.clear_contextvars()


def get_logger():
    """Return a structlog logger bound to the current context."""
    return structlog.get_logger()

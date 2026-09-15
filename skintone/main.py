"""Application factory and uvicorn entry point (contract sections 2.1, 2.6).

Responsibilities that must apply uniformly, hence live here rather than in a
route:

* JSON logging (``python-json-logger``).
* CORS from ``SKINTONE_ALLOWED_ORIGINS``, allowing the ``X-API-Key`` header.
* API-key authentication for everything except ``/v1/health``.
* Per-IP token-bucket rate limiting (``SKINTONE_RATE_LIMIT``).
* The single error envelope ``{"error": {...}}`` for every failure.
* Startup purge of archived items older than ``SKINTONE_RETENTION_DAYS``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, analysis, api, config, quota, schemas, storage

LOGGER = logging.getLogger("skintone")


def configure_logging(level: str) -> None:
    """Install a single JSON-lines log handler on the root logger.

    Args:
        level: Log level name, e.g. ``"INFO"``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "time", "levelname": "level", "name": "logger"},
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Purge expired archive entries on startup, then serve."""
    store: storage.Storage = app.state.storage
    removed = store.purge_expired()
    LOGGER.info(
        "startup retention purge complete",
        extra={"removed": removed, "retentionDays": store.retention_days},
    )
    yield


def create_app(settings: config.Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Runtime settings; defaults to the process-wide
            :func:`skintone.config.get_settings`.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    resolved = settings or config.get_settings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title="skintone-serve",
        version=__version__,
        description=(
            "可复现的肤色测量后端：ITA 深浅、冷暖/橄榄底色、表面状态。"
            "接口以 CONTRACT.md v1.0.0 为准。"
        ),
        lifespan=_lifespan,
    )
    app.state.settings = resolved
    app.state.storage = storage.Storage(resolved.data_dir, resolved.retention_days)
    app.state.started_at = time.monotonic()
    # 每日配额：防滥用，不是认证。指针可伪造，只求挡住随手刷与填满磁盘。
    app.state.quota = quota.DailyQuota(
        resolved.quota_db_path,
        resolved.quota_per_day_client,
        resolved.quota_per_day_ip,
    )
    app.state.rate_limiter = api.RateLimiter(
        resolved.rate_limit_capacity, resolved.rate_limit_per_second
    )

    @app.exception_handler(schemas.ApiError)
    async def _api_error_handler(_: Request, error: schemas.ApiError) -> JSONResponse:
        """Render :class:`skintone.schemas.ApiError` as the contract envelope."""
        return _error_response(error)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, error: RequestValidationError) -> JSONResponse:
        """Render FastAPI's own validation failures in the contract envelope."""
        return _error_response(
            schemas.ApiError(
                "BAD_IMAGE",
                "请求参数不合法",
                "analyze 需要 multipart 字段 image（文件）与 meta（JSON 文本）",
                details={"errors": [str(item.get("msg", item)) for item in error.errors()]},
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(_: Request, error: StarletteHTTPException) -> JSONResponse:
        """Render framework HTTP errors in the contract envelope."""
        code = "UNAUTHORIZED" if error.status_code == 401 else "BAD_IMAGE"
        if error.status_code == 404:
            code = "CARD_PROFILE_NOT_FOUND"
        elif error.status_code == 429:
            code = "RATE_LIMITED"
        elif error.status_code >= 500:
            code = "INTERNAL"
        return _error_response(
            schemas.ApiError(code, str(error.detail), "", status_code=error.status_code)
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, error: Exception) -> JSONResponse:
        """Log and render unexpected failures (never leak a stack trace)."""
        LOGGER.exception("unhandled error", exc_info=error)
        return _error_response(
            schemas.ApiError(
                "INTERNAL", "服务端处理失败", "请查看服务端日志；这是缺陷，请上报"
            )
        )

    app.include_router(api.router)

    @app.middleware("http")
    async def _guard(request: Request, call_next: Any) -> Any:
        """Enforce size ceiling, rate limit, optional API key and the daily quota."""
        path = request.url.path
        counts_toward_quota = (
            resolved.quota_enabled
            and path.startswith("/v1/")
            and quota.is_quota_target(request.method, path)
        )
        try:
            if path.startswith("/v1/") and request.method != "OPTIONS":
                api.parse_size_guard(request, resolved)
                if path not in api.PUBLIC_PATHS:
                    allowed, retry_after = app.state.rate_limiter.check(api.client_key(request))
                    if not allowed:
                        raise schemas.ApiError(
                            "RATE_LIMITED",
                            f"请求过于频繁，已超过 {resolved.rate_limit} 的限制",
                            f"请在 {retry_after:.0f} 秒后重试",
                            details={"retryAfterSeconds": round(retry_after, 1)},
                        )
                    if resolved.auth_required:
                        supplied = request.headers.get("x-api-key", "")
                        if supplied != resolved.api_key:
                            raise schemas.ApiError(
                                "UNAUTHORIZED",
                                "缺少或不匹配的 X-API-Key",
                                "在请求头里带上服务端 .env 中配置的 SKINTONE_API_KEY",
                            )
                    if counts_toward_quota:
                        info = app.state.quota.peek(
                            quota.client_identifier(request), quota.client_ip(request)
                        )
                        request.state.quota_info = info
                        if info["remaining"] <= 0:
                            raise schemas.ApiError(
                                "RATE_LIMITED",
                                f"今天的 {info['clientLimit']} 次额度已经用完了",
                                "明天自动恢复。额度只在对上脸、真正出结果之后才扣，"
                                "所以失败的那几次不算数",
                                details=info,
                            )
        except schemas.ApiError as error:
            response = _error_response(error)
        else:
            response = await call_next(request)
            # 只在真正处理成功后才扣额度：拍糊了、没对上脸，不该白扣一次
            if counts_toward_quota and response.status_code < 400:
                info = app.state.quota.consume(
                    quota.client_identifier(request), quota.client_ip(request)
                )
                request.state.quota_info = info
        # 把剩余额度回给前端，让用户看得到自己还能测几次
        info = getattr(request.state, "quota_info", None)
        if info:
            for name, value in quota.quota_headers(info).items():
                response.headers[name] = value
        return response

    # Added last on purpose. Starlette wraps ``user_middleware`` in reverse, so the
    # most recently added middleware ends up outermost -- which is what keeps CORS
    # headers on the 401/413/429 responses produced by ``_guard`` above.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    LOGGER.info(
        "skintone-serve ready",
        extra={
            "specVersion": resolved.spec_version,
            "dataDir": str(resolved.data_dir),
            "authRequired": resolved.auth_required,
            "rateLimit": resolved.rate_limit,
            "quotaPerDayClient": resolved.quota_per_day_client,
            "quotaPerDayIp": resolved.quota_per_day_ip,
        },
    )
    return app


def _error_response(error: schemas.ApiError) -> JSONResponse:
    """Serialise an :class:`skintone.schemas.ApiError` to a JSON response."""
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_response().model_dump(),
    )


def main() -> None:
    """Run the service with uvicorn, reading host/port from the environment.

    The app is built through the factory (``--factory``) rather than at import
    time, so merely importing this module never creates a data directory.
    """
    import uvicorn

    settings = config.get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()

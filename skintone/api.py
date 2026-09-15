"""HTTP routes (contract section 2).

The router here is deliberately thin: parse multipart parts, call
:mod:`skintone.analysis`, shape the JSON. Authentication, rate limiting, CORS and
error rendering are attached in :mod:`skintone.main` so they apply uniformly.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from . import analysis, config, schemas, storage
from .cards import definitions

router = APIRouter(prefix="/v1")

#: Endpoints that never require an API key.
PUBLIC_PATHS = frozenset({"/v1/health"})


class RateLimiter:
    """Per-IP token bucket.

    The bucket size and refill rate come from ``SKINTONE_RATE_LIMIT`` (default
    ``30/minute``), parsed once in :mod:`skintone.config`.
    """

    def __init__(self, capacity: int, per_second: float) -> None:
        """Create a limiter with a burst ``capacity`` and ``per_second`` refill."""
        self.capacity = float(capacity)
        self.per_second = float(per_second)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Consume one token for ``key``.

        Args:
            key: Bucket key, normally the client IP.
            now: Monotonic timestamp override (for tests).

        Returns:
            ``(allowed, retry_after_seconds)``; ``retry_after`` is 0 when allowed.
        """
        moment = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, moment))
            tokens = min(self.capacity, tokens + (moment - last) * self.per_second)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, moment)
                return True, 0.0
            self._buckets[key] = (tokens, moment)
            deficit = 1.0 - tokens
            retry = deficit / self.per_second if self.per_second > 0 else 60.0
            return False, max(retry, 1.0)


def client_key(request: Request) -> str:
    """Return the rate-limit bucket key for a request (client IP, or ``"local"``)."""
    return request.client.host if request.client else "local"


def _parse_meta(raw: str) -> schemas.AnalyzeMeta:
    """Parse and validate the ``meta`` multipart part.

    Args:
        raw: Raw JSON text.

    Returns:
        The validated metadata.

    Raises:
        schemas.ApiError: ``BAD_IMAGE`` (the only generic client-input code the
            contract defines) with a precisely worded message.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise schemas.ApiError(
            "BAD_IMAGE",
            f"meta 不是合法 JSON：{error}",
            "meta 字段必须是 JSON 字符串，例如 {\"mode\":\"nocard\"}",
        ) from error
    if not isinstance(payload, dict):
        raise schemas.ApiError("BAD_IMAGE", "meta 必须是 JSON 对象", "请检查前端构造的 meta")
    try:
        return schemas.AnalyzeMeta.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - pydantic ValidationError
        raise schemas.ApiError(
            "BAD_IMAGE",
            f"meta 字段不合法：{error}",
            "mode 必须是 nocard / card / calibrate；capture.illuminantGuess 见契约 2.3",
        ) from error


def _parse_rois(raw: str | None) -> schemas.RoisSpec | None:
    """Parse the optional ``rois`` multipart part."""
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise schemas.ApiError(
            "BAD_IMAGE",
            f"rois 不是合法 JSON：{error}",
            "rois 必须是 JSON 字符串；坐标为原图像素",
        ) from error
    if not isinstance(payload, dict):
        raise schemas.ApiError("BAD_IMAGE", "rois 必须是 JSON 对象", "请检查前端构造的 rois")
    try:
        return schemas.RoisSpec.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - pydantic ValidationError
        raise schemas.ApiError(
            "BAD_IMAGE",
            f"rois 字段不合法：{error}",
            "每个区域需要 {\"label\": str, \"points\": [[x, y], ...]}",
        ) from error


def _read_upload(upload: UploadFile, settings: config.Settings) -> bytes:
    """Read an upload while enforcing the size ceiling.

    Args:
        upload: The multipart file part.
        settings: Runtime settings (``SKINTONE_MAX_UPLOAD_MB``).

    Returns:
        The raw bytes.

    Raises:
        schemas.ApiError: ``PAYLOAD_TOO_LARGE`` when the file exceeds the limit.
    """
    data = upload.file.read()
    if len(data) > settings.max_upload_bytes:
        raise schemas.ApiError(
            "PAYLOAD_TOO_LARGE",
            f"图片 {len(data) / 1048576:.1f} MB 超过上限 {settings.max_upload_mb} MB",
            "请在前端缩放后再上传，或调高 SKINTONE_MAX_UPLOAD_MB",
        )
    return data


@router.get("/health", response_model=schemas.HealthResponse)
def health(request: Request) -> schemas.HealthResponse:
    """Liveness probe. Never requires authentication (contract section 2.1)."""
    settings: config.Settings = request.app.state.settings
    return schemas.HealthResponse(
        status="ok",
        version=settings.server_version,
        specVersion=settings.spec_version,
        uptimeSeconds=analysis.uptime_seconds(request.app.state.started_at),
        cards=definitions.card_ids(),
        authRequired=settings.auth_required,
    )


@router.get("/card/{card_id}", response_model=schemas.CardSpec)
def get_card(card_id: str) -> schemas.CardSpec:
    """Return the printable specification of a reference card.

    Args:
        card_id: Card identifier.

    Raises:
        schemas.ApiError: 404 when the id is unknown. The contract's error-code
            list has no generic NOT_FOUND, so ``CARD_PROFILE_NOT_FOUND`` is used.
    """
    spec = definitions.card_spec(card_id)
    if spec is None:
        raise schemas.ApiError(
            "CARD_PROFILE_NOT_FOUND",
            f"未知的色卡 id：{card_id}",
            f"可用色卡：{', '.join(definitions.card_ids())}",
            status_code=404,
        )
    return schemas.CardSpec.model_validate(spec)


@router.post("/analyze", response_model=schemas.AnalyzeResponse)
def analyze(
    request: Request,
    image: UploadFile = File(...),
    meta: str = Form(...),
    rois: str | None = Form(default=None),
) -> schemas.AnalyzeResponse:
    """Analyse one photo and return the contract's full result document."""
    settings: config.Settings = request.app.state.settings
    store: storage.Storage = request.app.state.storage
    parsed_meta = _parse_meta(meta)
    parsed_rois = _parse_rois(rois)
    image_bytes = _read_upload(image, settings)

    payload, image_to_store = analysis.analyze(
        image_bytes,
        parsed_meta,
        parsed_rois,
        profile_lookup=store.get_profile,
    )
    request_id = analysis.new_request_id()
    payload["requestId"] = request_id
    # Validate first, then archive the *serialised* document so what is stored is
    # byte-for-byte what the client received from GET /v1/result/{id}.
    document = schemas.AnalyzeResponse.model_validate(payload)
    store.save_result(
        request_id=request_id,
        created_at=datetime.now(timezone.utc),
        mode=parsed_meta.mode,
        payload=document.model_dump(),
        image_bytes=image_to_store,
    )
    return document


@router.post(
    "/card/{card_id}/calibrate",
    response_model=schemas.CalibrateResponse,
)
def calibrate_card(
    card_id: str,
    request: Request,
    image: UploadFile = File(...),
    meta: str = Form(...),
) -> schemas.CalibrateResponse:
    """Self-calibrate a printed card: recover the colours this sheet really has."""
    settings: config.Settings = request.app.state.settings
    store: storage.Storage = request.app.state.storage
    parsed_meta = _parse_meta(meta)
    image_bytes = _read_upload(image, settings)
    image_bgr = analysis.decode_image(image_bytes)

    if card_id not in definitions.CARD_IDS:
        raise schemas.ApiError(
            "CARD_NOT_DETECTED",
            f"未知的色卡 id：{card_id}",
            f"可用色卡：{', '.join(definitions.card_ids())}",
        )
    parsed_meta.cardId = card_id

    now = datetime.now(timezone.utc)
    response, profile = analysis.calibrate_card(
        image_bgr, parsed_meta, parsed_meta.cardProfileId or analysis.new_request_id(), now
    )
    store.save_profile(response["profileId"], card_id, now, profile)
    return schemas.CalibrateResponse.model_validate(response)


@router.get("/result/{request_id}")
def get_result(request_id: str, request: Request) -> JSONResponse:
    """Return a stored, de-identified result. The original image is never returned."""
    store: storage.Storage = request.app.state.storage
    stored = store.get_result(request_id)
    if stored is None:
        raise schemas.ApiError(
            "CARD_PROFILE_NOT_FOUND",
            f"未找到结果 {request_id}",
            "结果可能已被删除或超过保留期",
            status_code=404,
        )
    return JSONResponse(content=stored.payload)


@router.delete("/result/{request_id}", status_code=204)
def delete_result(request_id: str, request: Request) -> Response:
    """Delete a result: metadata row **and** archived original image."""
    store: storage.Storage = request.app.state.storage
    if not store.delete_result(request_id):
        raise schemas.ApiError(
            "CARD_PROFILE_NOT_FOUND",
            f"未找到结果 {request_id}",
            "结果可能已被删除或超过保留期",
            status_code=404,
        )
    return Response(status_code=204)


@router.get("/stats", response_model=schemas.StatsResponse)
def stats(request: Request) -> schemas.StatsResponse:
    """Return aggregate counters only (contract section 2.6)."""
    store: storage.Storage = request.app.state.storage
    return schemas.StatsResponse.model_validate(store.stats())


def parse_size_guard(request: Request, settings: config.Settings) -> None:
    """Reject an oversized body before multipart parsing starts.

    Args:
        request: Incoming request.
        settings: Runtime settings.

    Raises:
        schemas.ApiError: ``PAYLOAD_TOO_LARGE`` when ``Content-Length`` exceeds the
            configured ceiling (with a little slack for the other form fields).
    """
    header = request.headers.get("content-length")
    if not header:
        return
    try:
        declared = int(header)
    except ValueError:
        return
    if declared > settings.max_upload_bytes + 1024 * 1024:
        raise schemas.ApiError(
            "PAYLOAD_TOO_LARGE",
            f"请求体 {declared / 1048576:.1f} MB 超过上限 {settings.max_upload_mb} MB",
            "请在前端缩放后再上传，或调高 SKINTONE_MAX_UPLOAD_MB",
        )

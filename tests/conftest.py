"""Shared pytest fixtures.

The application is exercised through Starlette's synchronous ``TestClient``,
which drives the real ASGI stack (routing, middleware, lifespan, multipart
parsing, response models) without opening a socket. One test in ``test_api.py``
additionally boots uvicorn for the contract's smoke check.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from skintone import config
from skintone.main import create_app


@pytest.fixture(autouse=True)
def _isolate_from_local_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """开发机上的 ``.env`` 不得影响测试结果。

    否则任何人把 ``.env.example`` 复制成 ``.env`` 并设了 ``SKINTONE_API_KEY``，
    套件就会因为认证被打开而挂掉（例如 ``test_payload_too_large`` 本该拿到
    413，却先被 401 拦住）。这是测试环境泄漏，不是被测代码的问题。

    pydantic-settings 的来源优先级是 init > 环境变量 > dotenv，
    所以在环境变量层面覆盖即可压过 ``.env``。
    """
    monkeypatch.setenv("SKINTONE_API_KEY", "")
    monkeypatch.setenv("SKINTONE_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("SKINTONE_DATA_DIR", str(tmp_path / "env-data"))


@pytest.fixture
def settings(tmp_path: Path) -> config.Settings:
    """Settings pointed at a throwaway data directory with a generous rate limit."""
    return config.Settings(
        data_dir=tmp_path / "data",
        rate_limit="1000/minute",
        api_key="",
        allowed_origins="*",
        retention_days=30,
    )


@pytest.fixture
def app(settings: config.Settings):
    """A freshly built FastAPI application."""
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    """A synchronous HTTP client bound to the ASGI app, with lifespan run."""
    with TestClient(app) as test_client:
        yield test_client


def post_analyze(
    client: TestClient,
    image_bytes: bytes,
    meta: dict,
    rois: dict | None = None,
    filename: str = "capture.jpg",
):
    """POST /v1/analyze with the contract's multipart fields."""
    import json

    files = {"image": (filename, image_bytes, "image/jpeg")}
    data = {"meta": json.dumps(meta)}
    if rois is not None:
        data["rois"] = json.dumps(rois)
    return client.post("/v1/analyze", files=files, data=data)

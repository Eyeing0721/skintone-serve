"""HTTP contract tests: health, card spec, auth, rate limiting, storage endpoints."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from skintone import config
from skintone.main import create_app
from tests import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_jpeg() -> bytes:
    """A small valid JPEG."""
    return synthetic.encode_jpeg(np.full((40, 40, 3), 150, dtype=np.uint8))


def _nocard_meta(store_image: bool = True) -> dict:
    """Minimal valid ``meta`` for a no-card request."""
    return {
        "mode": "nocard",
        "capture": {"wbLocked": True, "illuminantGuess": "unknown"},
        "consent": {"storeImage": store_image},
        "clientVersion": "0.1.0",
    }


def _square_rois(size: int = 40) -> dict:
    """A skin ROI covering most of the tiny test image."""
    return {
        "skin": [
            {
                "label": "jaw",
                "points": [[5, 5], [size - 5, 5], [size - 5, size - 5], [5, size - 5]],
            }
        ]
    }


def test_health_returns_the_contract_payload(client: TestClient) -> None:
    """``GET /v1/health`` must be unauthenticated and carry the spec version."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["specVersion"] == "1.0.0"
    assert body["version"] == "0.1.0"
    assert body["cards"] == ["skintone-a4-v1"]
    assert body["authRequired"] is False
    assert isinstance(body["uptimeSeconds"], (int, float))


def test_health_is_never_rate_limited(settings: config.Settings, tmp_path: Path) -> None:
    """Liveness probes must keep working when the limiter is exhausted."""
    strict = config.Settings(data_dir=tmp_path / "data", rate_limit="1/minute")
    with TestClient(create_app(strict)) as limited:
        for _ in range(5):
            assert limited.get("/v1/health").status_code == 200


def test_card_spec_matches_the_contract(client: TestClient) -> None:
    """``GET /v1/card/{id}`` must expose the full printable geometry."""
    response = client.get("/v1/card/skintone-a4-v1")
    assert response.status_code == 200
    body = response.json()
    assert body["cardId"] == "skintone-a4-v1"
    assert body["specVersion"] == "1.0.0"
    assert body["paper"]["dpi"] == 300
    assert body["markers"]["dictionary"] == "DICT_4X4_50"
    assert body["markers"]["ids"] == [0, 1, 2, 3]
    assert len(body["patches"]) == 30
    assert {patch["kind"] for patch in body["patches"]} == {"gray", "color", "skin", "olive"}


def test_unknown_card_returns_404(client: TestClient) -> None:
    """An unknown card id must be a 404 with the error envelope."""
    response = client.get("/v1/card/no-such-card")
    assert response.status_code == 404
    assert "error" in response.json()


def test_stats_endpoint(client: TestClient) -> None:
    """``GET /v1/stats`` returns counters only."""
    response = client.get("/v1/stats")
    assert response.status_code == 200
    assert response.json() == {"totalRequests": 0, "storedImages": 0, "diskBytes": 0}


def test_bad_image_is_rejected_with_the_contract_code(client: TestClient) -> None:
    """A non-image payload must produce ``BAD_IMAGE``."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", b"not an image", "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta())},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "BAD_IMAGE"
    assert body["error"]["message"]
    assert body["error"]["hint"]
    assert set(body["error"]) == {"code", "message", "hint", "details"}


def test_bad_meta_is_rejected(client: TestClient) -> None:
    """Malformed ``meta`` JSON must be reported, not crashed on."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", _tiny_jpeg(), "image/jpeg")},
        data={"meta": "{not json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_IMAGE"


def test_missing_image_field_is_rejected(client: TestClient) -> None:
    """The multipart contract requires both ``image`` and ``meta``."""
    response = client.post("/v1/analyze", data={"meta": json.dumps(_nocard_meta())})
    assert response.status_code == 400
    assert "error" in response.json()


def test_card_mode_without_a_card_reports_card_not_detected(client: TestClient) -> None:
    """No markers -> ``CARD_NOT_DETECTED`` with an actionable hint."""
    meta = _nocard_meta()
    meta["mode"] = "card"
    meta["cardId"] = "skintone-a4-v1"
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", _tiny_jpeg(), "image/jpeg")},
        data={"meta": json.dumps(meta), "rois": json.dumps(_square_rois())},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "CARD_NOT_DETECTED"
    assert "4" in body["error"]["message"]


def test_nocard_without_roi_and_without_a_face_reports_face_not_found(client: TestClient) -> None:
    """The Haar fallback must fail loudly rather than guess an ROI."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", _tiny_jpeg(), "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta())},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "FACE_NOT_FOUND"


def test_unknown_card_profile_is_reported(client: TestClient) -> None:
    """A ``cardProfileId`` that was never calibrated must be an error."""
    meta = _nocard_meta()
    meta["mode"] = "card"
    meta["cardId"] = "skintone-a4-v1"
    meta["cardProfileId"] = "never-calibrated"
    image = synthetic.encode_jpeg(synthetic.render_card())
    response = client.post(
        "/v1/analyze",
        files={"image": ("card.jpg", image, "image/jpeg")},
        data={"meta": json.dumps(meta), "rois": json.dumps({"skin": [{"label": "jaw", "points": [[10, 10], [60, 10], [60, 60], [10, 60]]}]})},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CARD_PROFILE_NOT_FOUND"


def test_result_round_trip_and_cascade_delete(app, settings: config.Settings, client: TestClient) -> None:
    """Store -> fetch -> delete, and the archived file must disappear."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", synthetic.encode_jpeg(np.full((60, 60, 3), 170, dtype=np.uint8)), "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta()), "rois": json.dumps(_square_rois(60))},
    )
    assert response.status_code == 200
    request_id = response.json()["requestId"]

    stored = app.state.storage
    archived = list(stored.archive_dir.rglob("*.jpg"))
    assert len(archived) == 1

    fetched = client.get(f"/v1/result/{request_id}")
    assert fetched.status_code == 200
    assert fetched.json()["requestId"] == request_id

    deleted = client.delete(f"/v1/result/{request_id}")
    assert deleted.status_code == 204
    assert not list(stored.archive_dir.rglob("*.jpg"))
    assert client.get(f"/v1/result/{request_id}").status_code == 404
    assert client.delete(f"/v1/result/{request_id}").status_code == 404


def test_store_image_false_writes_no_file(app, client: TestClient) -> None:
    """Without consent nothing may reach the disk."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", synthetic.encode_jpeg(np.full((60, 60, 3), 170, dtype=np.uint8)), "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta(store_image=False)), "rois": json.dumps(_square_rois(60))},
    )
    assert response.status_code == 200
    assert not list(app.state.storage.archive_dir.rglob("*.jpg"))
    assert app.state.storage.stats()["storedImages"] == 0


def test_payload_too_large(tmp_path: Path) -> None:
    """An oversized upload must be refused with ``PAYLOAD_TOO_LARGE`` / 413."""
    strict = config.Settings(data_dir=tmp_path / "data", max_upload_mb=1, rate_limit="1000/minute")
    rng = np.random.default_rng(3)
    big = rng.integers(0, 255, (700, 700, 3), dtype=np.uint8)
    payload = synthetic.encode_png(big)
    assert len(payload) > 1024 * 1024
    with TestClient(create_app(strict)) as limited:
        response = limited.post(
            "/v1/analyze",
            files={"image": ("big.png", payload, "image/png")},
            data={"meta": json.dumps(_nocard_meta())},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_authentication_is_enforced(tmp_path: Path) -> None:
    """With an API key configured everything except /v1/health needs the header."""
    secured = config.Settings(
        data_dir=tmp_path / "data", api_key="s3cret", rate_limit="1000/minute"
    )
    with TestClient(create_app(secured)) as secured_client:
        health = secured_client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["authRequired"] is True

        assert secured_client.get("/v1/stats").status_code == 401
        assert secured_client.get("/v1/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        unauthorized = secured_client.get("/v1/stats", headers={"X-API-Key": "nope"})
        assert unauthorized.json()["error"]["code"] == "UNAUTHORIZED"
        assert secured_client.get("/v1/stats", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    """Exceeding the token bucket must yield 429 in the contract envelope."""
    limited_settings = config.Settings(
        data_dir=tmp_path / "data", rate_limit="2/minute", api_key=""
    )
    with TestClient(create_app(limited_settings)) as limited:
        assert limited.get("/v1/stats").status_code == 200
        assert limited.get("/v1/stats").status_code == 200
        blocked = limited.get("/v1/stats")
        assert blocked.status_code == 429
        body = blocked.json()
        assert body["error"]["code"] == "RATE_LIMITED"
        assert "retryAfterSeconds" in body["error"]["details"]
        # Health stays reachable even when the bucket is empty.
        assert limited.get("/v1/health").status_code == 200


def test_cors_headers_are_present(client: TestClient) -> None:
    """Browsers must be allowed to send ``X-API-Key`` and read the response."""
    response = client.options(
        "/v1/analyze",
        headers={
            "Origin": "https://example.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert response.status_code in (200, 204)
    allow_headers = response.headers.get("access-control-allow-headers", "").lower()
    assert "x-api-key" in allow_headers or "*" in allow_headers


def test_illuminant_prior_is_reported_as_applied(client: TestClient) -> None:
    """"照片光源"选了具体值时，响应必须承认先验确实生效了。

    回归防线：schema 里 locus / priorApplied / priorWeight 都带默认值，
    所以装配处漏了透传**不会报错**，契约键测试也照样通过。只有这条断言
    priorApplied 真正为 True 的测试才拦得住（实测中确实漏过一次）。
    """
    meta = _nocard_meta()
    meta["capture"] = {**meta["capture"], "illuminantGuess": "tungsten"}
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", synthetic.encode_jpeg(np.full((60, 60, 3), 170, dtype=np.uint8)), "image/jpeg")},
        data={"meta": json.dumps(meta), "rois": json.dumps(_square_rois(60))},
    )
    assert response.status_code == 200, response.text
    light = response.json()["illuminant"]
    assert light["priorApplied"] is True
    assert light["priorWeight"] == 0.25
    assert light["locus"] == "planckian"  # 白炽灯挂在普朗克轨迹上


def test_illuminant_prior_is_reported_as_ignored_when_unknown(client: TestClient) -> None:
    """选"不确定"时必须如实报告"先验没参与"，不能假装用了。"""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", synthetic.encode_jpeg(np.full((60, 60, 3), 170, dtype=np.uint8)), "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta()), "rois": json.dumps(_square_rois(60))},
    )
    assert response.status_code == 200, response.text
    light = response.json()["illuminant"]
    assert light["priorApplied"] is False
    assert light["priorWeight"] == 0.0


def test_cors_headers_apply_to_error_responses(tmp_path: Path) -> None:
    """The guard must sit *inside* CORS so 401s are still browser-readable."""
    secured = config.Settings(data_dir=tmp_path / "data", api_key="k", rate_limit="1000/minute")
    with TestClient(create_app(secured)) as secured_client:
        response = secured_client.get("/v1/stats", headers={"Origin": "https://example.github.io"})
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") in {"*", "https://example.github.io"}


def test_response_contains_every_contract_key(client: TestClient) -> None:
    """No key may be omitted; unmeasurable values must be present as null."""
    response = client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", synthetic.encode_jpeg(np.full((60, 60, 3), 170, dtype=np.uint8)), "image/jpeg")},
        data={"meta": json.dumps(_nocard_meta()), "rois": json.dumps(_square_rois(60))},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "requestId",
        "specVersion",
        "serverVersion",
        "createdAt",
        "mode",
        "confidence",
        "illuminant",
        "calibration",
        "skin",
        "advice",
        "warnings",
        "disclaimer",
    }
    assert body["calibration"] is None  # no card in this request
    assert set(body["illuminant"]) == {
        "method",
        "cct",
        "duv",
        "xy",
        "adaptation",
        "assumedD65",
        # 契约 §2.3 的扩展字段：让"用户选的光源有没有生效"可查，
        # 而不是一个黑盒下拉框（见 tests/test_illuminant.py 的回归防线）。
        "locus",
        "priorApplied",
        "priorWeight",
    }
    assert set(body["skin"]) == {
        "roi",
        "labD65",
        "linearRgb",
        "hex",
        "itaDeg",
        "depthClass",
        "hueAngleDeg",
        "chroma",
        "undertone",
        "melaninIndex",
        "hemoglobinIndex",
    }


def _free_port() -> int:
    """Return a currently unused TCP port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_uvicorn_smoke_health(tmp_path: Path) -> None:
    """Acceptance 9.2: boot uvicorn for real and read ``GET /v1/health``.

    This is the only test that goes over a TCP socket, and it is the exact smoke
    check the contract asks for.
    """
    import httpx

    port = _free_port()
    environment = dict(os.environ)
    environment.update(
        {
            "SKINTONE_DATA_DIR": str(tmp_path / "data"),
            "SKINTONE_PORT": str(port),
            "SKINTONE_HOST": "127.0.0.1",
            "SKINTONE_LOG_LEVEL": "WARNING",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "skintone.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(trust_env=False, timeout=5.0) as http:
            body = None
            for _ in range(60):
                try:
                    response = http.get(f"http://127.0.0.1:{port}/v1/health")
                    if response.status_code == 200:
                        body = response.json()
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            assert body is not None, "uvicorn did not answer /v1/health in time"
            assert body["specVersion"] == "1.0.0"
            assert body["status"] == "ok"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()

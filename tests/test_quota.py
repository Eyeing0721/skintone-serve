"""每日配额：存储层行为 + 挂进中间件后的 HTTP 行为。

配额是防滥用而不是认证，所以这里测的是"拦得住随手刷"和"别误伤正常用户"
（失败不扣额度、被拒不额外扣额度）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from skintone import config, quota
from skintone.main import create_app
from tests import synthetic


class _FakeRequest:
    """最小可用的 Request 替身，只提供 quota 模块用到的那两个属性。"""

    def __init__(self, headers: dict[str, str], host: str = "9.9.9.9") -> None:
        self.headers = headers
        self.client = type("Client", (), {"host": host})()


def make_store(tmp_path: Path, per_client: int = 2, per_ip: int = 3) -> quota.DailyQuota:
    return quota.DailyQuota(tmp_path / "quota.sqlite", per_client, per_ip)


def build_app(tmp_path: Path, **overrides):
    options = {
        "data_dir": tmp_path / "data",
        "rate_limit": "1000/minute",
        "api_key": "",
        "allowed_origins": "*",
        "quota_enabled": True,
        "quota_per_day_client": 2,
        "quota_per_day_ip": 10,
    }
    options.update(overrides)
    return create_app(config.Settings(**options))


def post_analyze(client: TestClient, client_id: str = "abc"):
    """发一个注定失败的分析请求（内容不是真图）。失败不该扣额度。"""
    return client.post(
        "/v1/analyze",
        files={"image": ("x.jpg", b"\xff\xd8\xff-not-a-real-jpeg", "image/jpeg")},
        data={"meta": "{}"},
        headers={"X-Client-Id": client_id},
    )


# ── 存储层 ───────────────────────────────────────────────────────────
def test_client_limit_then_blocked(tmp_path):
    store = make_store(tmp_path, per_client=2, per_ip=10)
    first = store.consume("c:aaa", "1.1.1.1")
    second = store.consume("c:aaa", "1.1.1.1")
    third = store.consume("c:aaa", "1.1.1.1")
    assert first["allowed"] and first["remaining"] == 1
    assert second["allowed"] and second["remaining"] == 0
    assert not third["allowed"] and third["blockedBy"] == "client"


def test_ip_limit_blocks_another_browser(tmp_path):
    store = make_store(tmp_path, per_client=10, per_ip=2)
    assert store.consume("c:a", "2.2.2.2")["allowed"]
    assert store.consume("c:b", "2.2.2.2")["allowed"]
    blocked = store.consume("c:c", "2.2.2.2")
    assert not blocked["allowed"] and blocked["blockedBy"] == "ip"


def test_refused_requests_do_not_burn_quota(tmp_path):
    store = make_store(tmp_path, per_client=1, per_ip=10)
    store.consume("c:a", "3.3.3.3")
    for _ in range(3):
        assert not store.consume("c:a", "3.3.3.3")["allowed"]
    assert store.peek("c:a", "3.3.3.3")["clientUsed"] == 1


def test_peek_never_consumes(tmp_path):
    store = make_store(tmp_path)
    for _ in range(3):
        store.peek("c:a", "4.4.4.4")
    assert store.peek("c:a", "4.4.4.4")["clientUsed"] == 0


def test_zero_limit_disables_that_scope(tmp_path):
    store = make_store(tmp_path, per_client=0, per_ip=10)
    result = store.consume("c:a", "5.5.5.5")
    assert not result["allowed"] and result["blockedBy"] == "client"


def test_counters_survive_a_new_instance(tmp_path):
    make_store(tmp_path, per_client=2, per_ip=10).consume("c:a", "6.6.6.6")
    reopened = make_store(tmp_path, per_client=2, per_ip=10)
    assert reopened.peek("c:a", "6.6.6.6")["clientUsed"] == 1


def test_purge_removes_older_days(tmp_path):
    store = make_store(tmp_path)
    store.consume("c:a", "7.7.7.7")
    assert store.purge_before("9999-01-01") == 2  # client + ip 两行
    assert store.peek("c:a", "7.7.7.7")["clientUsed"] == 0


# ── 请求解析 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/analyze", True),
        ("POST", "/v1/analyze/", True),
        ("POST", "/v1/card/skintone-a4-v1/calibrate", True),
        ("POST", "/v1/card/skintone-a4-v1/calibrate/", True),
        ("GET", "/v1/analyze", False),
        ("GET", "/v1/health", False),
        ("POST", "/v1/health", False),
        ("DELETE", "/v1/result/abc", False),
        ("GET", "/v1/card/skintone-a4-v1", False),
    ],
)
def test_is_quota_target(method, path, expected):
    assert quota.is_quota_target(method, path) is expected


def test_client_ip_prefers_the_tunnel_header():
    # 走 cloudflared 隧道时 socket 对端永远是本机，真实 IP 只在 CF 头里
    assert quota.client_ip(_FakeRequest({"cf-connecting-ip": "8.8.8.8"})) == "8.8.8.8"
    assert quota.client_ip(_FakeRequest({"x-forwarded-for": "7.7.7.7, 6.6.6.6"})) == "7.7.7.7"
    assert quota.client_ip(_FakeRequest({})) == "9.9.9.9"


def test_client_identifier_is_stable_and_hashed():
    ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc123"}))
    assert ident == quota.client_identifier(_FakeRequest({"x-client-id": "abc123"}))
    assert "abc123" not in ident  # 不存原始指纹，只需要相等性
    assert ident.startswith("c:")
    # 没有指纹头时退回 User-Agent，仍然稳定
    ua = _FakeRequest({"user-agent": "SomeBrowser/1.0"})
    assert quota.client_identifier(ua) == quota.client_identifier(ua)


# ── HTTP 层 ──────────────────────────────────────────────────────────
def test_failed_analysis_does_not_burn_quota(tmp_path):
    """拍糊/没对上脸不扣额度——5 次限制下这一点很关键。"""
    app = build_app(tmp_path, quota_per_day_client=5, quota_per_day_ip=50)
    with TestClient(app) as client:
        for _ in range(6):
            assert post_analyze(client).status_code >= 400
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        assert app.state.quota.peek(ident, "testclient")["clientUsed"] == 0


def test_http_returns_429_once_quota_is_exhausted(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=1, quota_per_day_ip=50)
    with TestClient(app) as client:
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        app.state.quota.consume(ident, "testclient")  # 手动用满这唯一一次
        res = post_analyze(client)
        assert res.status_code == 429
        assert res.json()["error"]["code"] == "RATE_LIMITED"
        assert res.headers["X-Quota-Remaining"] == "0"
        assert res.headers["X-Quota-Limit"] == "1"


def test_another_browser_still_allowed_when_only_one_is_exhausted(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=1, quota_per_day_ip=50)
    with TestClient(app) as client:
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        app.state.quota.consume(ident, "testclient")
        # 换个指纹，但 IP 没满（上限 50），应当不再被配额拦住
        res = post_analyze(client, client_id="different-browser")
        assert res.status_code != 429


def test_health_is_never_quota_blocked(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=1, quota_per_day_ip=1)
    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/v1/health").status_code == 200


@pytest.fixture(scope="module")
def nocard_capture():
    """一张肯定能测出结果的无卡合成图（复用验收测试的同一套合成手段）。

    模块级缓存：合成一张带噪声的场景不便宜，而这个用例只需一张。
    """
    scene, polygons, _info = synthetic.render_nocard_scene((58.30, 13.10, 19.70))
    lit = synthetic.simulate_illuminant(scene, synthetic.warm_illuminant_xy(), exposure=0.95)
    noisy = synthetic.add_sensor_noise(lit, sigma=1.0, seed=424242)
    return synthetic.encode_jpeg(noisy, quality=92), polygons


def test_successful_analysis_consumes_exactly_one(tmp_path, nocard_capture):
    """只有真正出结果才扣一次，并把余量通过响应头回给前端。

    这是配额的核心语义：失败的请求不扣（见上），成功的请求必须扣且只扣一次。
    """
    image, polygons = nocard_capture
    app = build_app(tmp_path, quota_per_day_client=4, quota_per_day_ip=40)
    meta = {
        "mode": "nocard",
        "capture": {
            "wbLocked": False, "raw": False, "flash": False,
            "illuminantGuess": "daylight", "devicePixelRatio": 1,
        },
        "consent": {"storeImage": False, "acceptedAt": "2026-09-15T00:00:00Z"},
        "clientVersion": "0.1.0",
    }
    with TestClient(app) as client:
        res = client.post(
            "/v1/analyze",
            files={"image": ("capture.jpg", image, "image/jpeg")},
            data={
                "meta": json.dumps(meta),
                "rois": json.dumps({"skin": [{"label": "jaw", "points": polygons[0]}]}),
            },
            headers={"X-Client-Id": "abc"},
        )
        assert res.status_code == 200, res.text
        assert res.headers["x-quota-remaining"] == "3"
        assert res.headers["x-quota-limit"] == "4"
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        assert app.state.quota.peek(ident, "testclient")["clientUsed"] == 1

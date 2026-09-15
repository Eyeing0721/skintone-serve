"""防滥用闸门：存储层行为 + 挂进中间件后的 HTTP 行为。

这里测的是三件事：
  1. **不误伤真人** —— 失败不扣、被拒不额外扣、冷却只挡连点
  2. **拦得住随手刷** —— 单指纹上限、全站上限、字节预算
  3. **不依赖 IP** —— 用户明确要求去掉 IP（CGNAT 会误伤），所以有专门的回归防线
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from skintone import config, quota
from skintone.main import create_app
from tests import synthetic

MEGABYTE = 1024 * 1024


class _FakeRequest:
    """最小可用的 Request 替身，只提供 quota 模块用到的那两个属性。"""

    def __init__(self, headers: dict[str, str], host: str = "9.9.9.9") -> None:
        self.headers = headers
        self.client = type("Client", (), {"host": host})()


def make_store(
    tmp_path: Path,
    per_client: int = 2,
    cooldown: float = 0.0,
    global_per_day: int = 100,
    global_mb: float = 1000.0,
) -> quota.GuardStore:
    return quota.GuardStore(
        tmp_path / "guard.sqlite",
        per_client_per_day=per_client,
        client_cooldown_seconds=cooldown,
        global_per_day=global_per_day,
        global_mb_per_day=global_mb,
    )


def build_app(tmp_path: Path, **overrides):
    options = {
        "data_dir": tmp_path / "data",
        "rate_limit": "1000/minute",
        "global_rate_limit": "1000/minute",
        "api_key": "",
        "allowed_origins": "*",
        "quota_enabled": True,
        "quota_per_day_client": 2,
        "quota_cooldown_seconds": 0.0,
        "quota_global_per_day": 100,
        "quota_global_mb_per_day": 1000.0,
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


# ── 回归防线：额度不得再依赖 IP ──────────────────────────────────────
def test_identifier_ignores_forwarding_headers():
    """同一指纹来自不同 IP 必须落进同一个桶——这正是去掉 IP 的目的。"""
    a = quota.client_identifier(
        _FakeRequest({"x-client-id": "same", "cf-connecting-ip": "1.1.1.1"})
    )
    b = quota.client_identifier(
        _FakeRequest({"x-client-id": "same", "cf-connecting-ip": "2.2.2.2"})
    )
    assert a == b


def test_quota_module_has_no_ip_entry_point():
    """配额模块里不该再有按 IP 计数的入口。"""
    assert not hasattr(quota, "client_ip")


# ── 存储层 ───────────────────────────────────────────────────────────
def test_client_limit_then_blocked(tmp_path):
    store = make_store(tmp_path, per_client=2)
    store.record_success("c:a")
    store.record_success("c:a")
    decision = store.check("c:a")
    assert not decision["allowed"] and decision["reason"] == "client_daily"
    assert decision["remaining"] == 0


def test_a_different_fingerprint_is_not_blocked_by_another(tmp_path):
    """指纹之间互不影响；全站上限没到就该放行。"""
    store = make_store(tmp_path, per_client=1)
    store.record_success("c:a")
    assert not store.check("c:a")["allowed"]
    assert store.check("c:b")["allowed"]


def test_cooldown_blocks_immediate_retry_then_expires(tmp_path):
    store = make_store(tmp_path, per_client=10, cooldown=15.0)
    store.record_success("c:a", now=1000.0)
    blocked = store.check("c:a", now=1003.0)
    assert not blocked["allowed"] and blocked["reason"] == "client_cooldown"
    assert blocked["retryAfterSeconds"] == pytest.approx(12.0)
    assert store.check("c:a", now=1016.0)["allowed"]


def test_global_daily_ceiling_stops_a_fresh_fingerprint(tmp_path):
    """换指纹也撞得到的天花板——这是指纹可被清除之后的最后一道。"""
    store = make_store(tmp_path, per_client=10, global_per_day=1)
    store.record_success("c:a")
    decision = store.check("c:fresh")
    assert not decision["allowed"] and decision["reason"] == "global_daily"


def test_global_byte_budget_stops_uploads(tmp_path):
    store = make_store(tmp_path, per_client=10, global_mb=0.001)  # 约 1 KB
    store.record_success("c:a", upload_bytes=2000)
    decision = store.check("c:b")
    assert not decision["allowed"] and decision["reason"] == "global_bytes"


def test_check_never_mutates(tmp_path):
    store = make_store(tmp_path, per_client=5)
    for _ in range(4):
        assert store.check("c:a")["allowed"]
    assert store.peek("c:a")["clientUsed"] == 0


def test_refused_requests_do_not_charge(tmp_path):
    store = make_store(tmp_path, per_client=1)
    store.record_success("c:a")
    store.check("c:a")  # 预检被拒
    assert store.peek("c:a")["clientUsed"] == 1


def test_record_success_accumulates_bytes(tmp_path):
    store = make_store(tmp_path)
    store.record_success("c:a", upload_bytes=1000)
    store.record_success("c:a", upload_bytes=500)
    assert store.peek("c:a")["bytesUsed"] == 1500


def test_zero_limit_disables_that_gate(tmp_path):
    store = make_store(tmp_path, per_client=0)
    assert store.check("c:a")["allowed"]  # 0 表示不限制


def test_counters_survive_a_new_instance(tmp_path):
    make_store(tmp_path, per_client=5).record_success("c:a")
    reopened = make_store(tmp_path, per_client=5)
    assert reopened.peek("c:a")["clientUsed"] == 1


def test_purge_removes_older_days(tmp_path):
    store = make_store(tmp_path)
    store.record_success("c:a", upload_bytes=10)
    assert store.purge_before("9999-01-01") == 3  # client + global + bytes
    assert store.peek("c:a")["clientUsed"] == 0


# ── 请求解析与文案 ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/v1/analyze", True),
        ("POST", "/v1/analyze/", True),
        ("POST", "/v1/card/skintone-a4-v1/calibrate", True),
        ("GET", "/v1/analyze", False),
        ("GET", "/v1/health", False),
        ("DELETE", "/v1/result/abc", False),
        ("GET", "/v1/card/skintone-a4-v1", False),
    ],
)
def test_is_quota_target(method, path, expected):
    assert quota.is_quota_target(method, path) is expected


def test_client_identifier_is_stable_and_hashed():
    ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc123"}))
    assert ident == quota.client_identifier(_FakeRequest({"x-client-id": "abc123"}))
    assert "abc123" not in ident  # 不存原始指纹，只需要相等性
    assert ident.startswith("c:")
    ua = _FakeRequest({"user-agent": "SomeBrowser/1.0"})
    assert quota.client_identifier(ua) == quota.client_identifier(ua)


@pytest.mark.parametrize(
    "reason", ["client_daily", "client_cooldown", "global_daily", "global_bytes"]
)
def test_refusal_wording_covers_every_reason(reason):
    """每种拒绝理由都要有可操作文案，且占位符必须被真正替换掉。"""
    message, hint = quota.describe_refusal(
        {
            "reason": reason,
            "clientLimit": 5,
            "globalLimit": 200,
            "bytesLimit": 500 * MEGABYTE,
            "retryAfterSeconds": 7.0,
            "cooldownSeconds": 15.0,
        }
    )
    assert message and hint
    assert "{" not in message and "{" not in hint


# ── HTTP 层 ──────────────────────────────────────────────────────────
def test_failed_analysis_does_not_burn_quota(tmp_path):
    """拍糊/没对上脸不扣额度——5 次限制下这一点很关键。"""
    app = build_app(tmp_path, quota_per_day_client=5)
    with TestClient(app) as client:
        for _ in range(6):
            assert post_analyze(client).status_code >= 400
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        assert app.state.guard.peek(ident)["clientUsed"] == 0


def test_http_returns_429_once_client_quota_is_exhausted(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=1)
    with TestClient(app) as client:
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        app.state.guard.record_success(ident)  # 手动用满这唯一一次
        res = post_analyze(client)
        assert res.status_code == 429
        assert res.json()["error"]["code"] == "RATE_LIMITED"
        assert res.headers["x-quota-remaining"] == "0"
        assert res.headers["x-quota-limit"] == "1"


def test_http_reports_cooldown_with_a_retry_hint(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=5, quota_cooldown_seconds=15.0)
    with TestClient(app) as client:
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        app.state.guard.record_success(ident)
        res = post_analyze(client)
        assert res.status_code == 429
        assert "秒后再试" in res.json()["error"]["message"]


def test_another_browser_still_allowed_when_only_one_is_exhausted(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=1)
    with TestClient(app) as client:
        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        app.state.guard.record_success(ident)
        assert post_analyze(client, client_id="different-browser").status_code != 429


def test_health_is_never_quota_blocked(tmp_path):
    app = build_app(tmp_path, quota_per_day_client=0, quota_global_per_day=0)
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


def _post_real(client: TestClient, image: bytes, polygons, client_id: str = "abc"):
    meta = {
        "mode": "nocard",
        "capture": {
            "wbLocked": False, "raw": False, "flash": False,
            "illuminantGuess": "daylight", "devicePixelRatio": 1,
        },
        "consent": {"storeImage": False, "acceptedAt": "2026-09-15T00:00:00Z"},
        "clientVersion": "0.1.0",
    }
    return client.post(
        "/v1/analyze",
        files={"image": ("capture.jpg", image, "image/jpeg")},
        data={
            "meta": json.dumps(meta),
            "rois": json.dumps({"skin": [{"label": "jaw", "points": polygons[0]}]}),
        },
        headers={"X-Client-Id": client_id},
    )


def test_successful_analysis_charges_exactly_one_and_reports_headers(tmp_path, nocard_capture):
    """只有真正出结果才扣一次，并把余量与上传字节都记上，同时回传响应头。"""
    image, polygons = nocard_capture
    app = build_app(tmp_path, quota_per_day_client=4)
    with TestClient(app) as client:
        res = _post_real(client, image, polygons)
        assert res.status_code == 200, res.text
        assert res.headers["x-quota-remaining"] == "3"
        assert res.headers["x-quota-limit"] == "4"

        ident = quota.client_identifier(_FakeRequest({"x-client-id": "abc"}))
        info = app.state.guard.peek(ident)
        assert info["clientUsed"] == 1
        # 记的是整个 multipart 请求体（含 meta/rois 与边界），所以比图片本身略大
        assert len(image) <= info["bytesUsed"] <= len(image) + 8192

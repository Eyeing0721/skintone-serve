"""配额与防滥用闸门 —— **不是认证**。

## 为什么里面没有 IP

手机运营商的 CGNAT 会让大量真实用户共用同一个出口 IP。按 IP 限流在写字楼、
校园网、地铁里必然误伤——一个人测完了，同网段的另一个人就被挡住。所以这里
**完全不使用客户端 IP**，改用一组"对真人零误伤"的闸门：

| 闸门 | 默认 | 作用 |
|---|---|---|
| 单指纹每日次数 | 5 | `X-Client-Id` 携带的浏览器指纹；可被清除重置，所以只是第一道 |
| 单指纹冷却 | 15 秒 | 真人不会十几秒内连拍两张，脚本会 |
| 全站每分钟令牌桶 | 120 | 不依赖任何标识，给洪水限速（在 main.py 里） |
| 全站每日次数 | 200 | 无论对方怎么换指纹都撞得到的硬天花板 |
| 全站每日上传字节 | 500 MB | 直接保护磁盘——这才是真正的风险 |
| 并发上限 | 2 | 保护 CPU（在 main.py 里用信号量） |

**它能防住什么**：随手刷的脚本、陌生人用你的机器跑批量测色、把磁盘写满。
**它防不住什么**：铁了心换 IP + 换指纹 + 慢速请求的人。那需要真正的认证
（`SKINTONE_API_KEY`）。这个取舍是有意的——宁可放过少数恶意者，也不误伤
任何一个真实用户。

## 记账时机

**只有成功出结果的请求才记账**（次数、字节、冷却都是）。拍糊了、没对上脸、
参数错误都不扣——5 次/天的额度下，这一点直接决定产品能不能用。
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: 换日边界用 UTC+8。用 UTC 换日会让用户在晚上 8 点后看到"明天"的配额。
LOCAL_TZ = timezone(timedelta(hours=8))

#: 只有这些请求计入配额。健康检查、取色卡规格、删除记录都不算。
QUOTA_METHOD = "POST"

#: 每个拒绝原因对应的可操作文案，集中在这里，别散到中间件里。
REASONS: dict[str, tuple[str, str]] = {
    "client_daily": (
        "今天的 {limit} 次额度已经用完了",
        "明天自动恢复。额度只在对上脸、真正出结果之后才扣，所以失败的那几次不算数",
    ),
    "client_cooldown": (
        "稍等一下，{wait} 秒后再试",
        "同一台设备两次测量之间需要间隔 {cooldown} 秒——这样也顺便避免了连点时拿到同一张糊图",
    ),
    "global_daily": (
        "今天全站的测量额度已经用完了",
        "这台机器每天只跑 {limit} 次，明天自动恢复",
    ),
    "global_bytes": (
        "今天全站的上传流量已经用完了",
        "这台机器每天只接收 {limit} MB，明天自动恢复",
    ),
}


def today_key() -> str:
    """Return the current local day as ``YYYY-MM-DD``."""
    return datetime.now(LOCAL_TZ).date().isoformat()


def client_identifier(request: Any) -> str:
    """Hash of the browser-fingerprint header, falling back to the User-Agent.

    Hashed so the raw fingerprint is never stored — only equality matters.
    No IP is used anywhere: see the module docstring.
    """
    raw = (request.headers.get("x-client-id") or "").strip()
    if not raw:
        raw = "ua:" + (request.headers.get("user-agent") or "")
    return "c:" + hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:32]


def is_quota_target(method: str, path: str) -> bool:
    """True when this request should consume one unit of quota."""
    if method.upper() != QUOTA_METHOD:
        return False
    clean = path.rstrip("/")
    return clean == "/v1/analyze" or clean.endswith("/calibrate")


def quota_headers(info: dict) -> dict[str, str]:
    """Response headers the client uses to show how many shots are left."""
    return {
        "X-Quota-Limit": str(info.get("clientLimit", "")),
        "X-Quota-Remaining": str(info.get("remaining", "")),
        "X-Quota-Day": str(info.get("day", "")),
    }


class GuardStore:
    """Daily counters plus a cooldown stamp, persisted in its own SQLite file.

    Scopes inside the ``counters`` table:

    * ``client`` — per-fingerprint attempt count for the day
    * ``global`` — site-wide attempt count for the day
    * ``bytes``  — site-wide upload bytes for the day
    """

    def __init__(
        self,
        db_path: Path,
        *,
        per_client_per_day: int,
        client_cooldown_seconds: float,
        global_per_day: int,
        global_mb_per_day: float,
    ) -> None:
        self._path = Path(db_path)
        self.per_client = max(0, int(per_client_per_day))
        self.cooldown = max(0.0, float(client_cooldown_seconds))
        self.global_per_day = max(0, int(global_per_day))
        self.global_bytes = max(0, int(float(global_mb_per_day) * 1024 * 1024))
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # -- internals ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    day   TEXT    NOT NULL,
                    scope TEXT    NOT NULL,
                    key   TEXT    NOT NULL,
                    used  INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, scope, key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cooldown (
                    key TEXT PRIMARY KEY,
                    ts  REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _read(conn: sqlite3.Connection, day: str, scope: str, key: str) -> int:
        row = conn.execute(
            "SELECT used FROM counters WHERE day = ? AND scope = ? AND key = ?",
            (day, scope, key),
        ).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _bump(conn: sqlite3.Connection, day: str, scope: str, key: str, amount: int) -> None:
        conn.execute(
            """
            INSERT INTO counters (day, scope, key, used) VALUES (?, ?, ?, ?)
            ON CONFLICT (day, scope, key) DO UPDATE SET used = used + excluded.used
            """,
            (day, scope, key, amount),
        )

    @staticmethod
    def _cooldown_ts(conn: sqlite3.Connection, key: str) -> float:
        row = conn.execute("SELECT ts FROM cooldown WHERE key = ?", (key,)).fetchone()
        return float(row[0]) if row else 0.0

    def _info(self, conn: sqlite3.Connection, day: str, ident: str) -> dict:
        client_used = self._read(conn, day, "client", ident)
        return {
            "day": day,
            "clientUsed": client_used,
            "clientLimit": self.per_client,
            "remaining": max(0, self.per_client - client_used),
            "globalUsed": self._read(conn, day, "global", "all"),
            "globalLimit": self.global_per_day,
            "bytesUsed": self._read(conn, day, "bytes", "all"),
            "bytesLimit": self.global_bytes,
        }

    # -- public API --------------------------------------------------------
    def peek(self, ident: str) -> dict:
        """Report usage without consuming or blocking."""
        day = today_key()
        with self._lock, self._connect() as conn:
            return self._info(conn, day, ident)

    def check(self, ident: str, now: float | None = None) -> dict:
        """Decide whether this request may proceed. Never mutates state."""
        moment = time.time() if now is None else now
        day = today_key()
        with self._lock, self._connect() as conn:
            info = self._info(conn, day, ident)
            last = self._cooldown_ts(conn, ident)

        if self.per_client and info["clientUsed"] >= self.per_client:
            return {**info, "allowed": False, "reason": "client_daily"}
        if self.cooldown and last and (moment - last) < self.cooldown:
            info["retryAfterSeconds"] = round(self.cooldown - (moment - last), 1)
            return {**info, "allowed": False, "reason": "client_cooldown"}
        if self.global_per_day and info["globalUsed"] >= self.global_per_day:
            return {**info, "allowed": False, "reason": "global_daily"}
        if self.global_bytes and info["bytesUsed"] >= self.global_bytes:
            return {**info, "allowed": False, "reason": "global_bytes"}
        return {**info, "allowed": True, "reason": None}

    def record_success(self, ident: str, upload_bytes: int = 0, now: float | None = None) -> dict:
        """Charge one unit after a request actually produced a result."""
        moment = time.time() if now is None else now
        day = today_key()
        with self._lock, self._connect() as conn:
            self._bump(conn, day, "client", ident, 1)
            self._bump(conn, day, "global", "all", 1)
            if upload_bytes > 0 and self.global_bytes:
                self._bump(conn, day, "bytes", "all", int(upload_bytes))
            conn.execute(
                "INSERT INTO cooldown (key, ts) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET ts = excluded.ts",
                (ident, moment),
            )
            conn.commit()
            return self._info(conn, day, ident)

    def purge_before(self, day: str) -> int:
        """Delete counters older than ``day``; returns rows removed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM counters WHERE day < ?", (day,))
            conn.commit()
            return int(cur.rowcount or 0)


def describe_refusal(info: dict) -> tuple[str, str]:
    """Turn a refusal into ``(message, hint)`` using :data:`REASONS`."""
    reason = info.get("reason") or "client_daily"
    template, hint_template = REASONS.get(reason, REASONS["client_daily"])
    values = {
        "limit": info.get("clientLimit") if reason != "global_daily" else info.get("globalLimit"),
        "wait": info.get("retryAfterSeconds", 0),
        "cooldown": info.get("cooldownSeconds", 0),
    }
    if reason == "global_bytes":
        values["limit"] = round((info.get("bytesLimit") or 0) / (1024 * 1024))
    return template.format(**values), hint_template.format(**values)

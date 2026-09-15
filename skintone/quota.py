"""Per-browser / per-IP daily quota.

**This is abuse protection, not authentication.** The browser fingerprint travels
in ``X-Client-Id`` and is trivially forgeable — anyone who can edit a request
header can reset it. That is an accepted trade-off: the goal is to stop casual
scripted abuse and stop strangers filling this machine's disk, not to keep
secrets. Real access control is the optional ``SKINTONE_API_KEY``.

Quota state lives in its own SQLite file so it survives restarts, and the "day"
rolls over in UTC+8 — the user-visible "today" has to be their local today.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: 以中国时区换日。用 UTC 换日会让用户在晚上 8 点后看到"明天"的配额。
LOCAL_TZ = timezone(timedelta(hours=8))

#: 只有这些请求消耗配额。健康检查、取色卡规格、删除记录都不算。
QUOTA_METHOD = "POST"


def today_key() -> str:
    """Return the current local day as ``YYYY-MM-DD``."""
    return datetime.now(LOCAL_TZ).date().isoformat()


def client_ip(request) -> str:
    """Best-effort client IP, honouring the tunnel's forwarding headers.

    Through a cloudflared tunnel the socket peer is always localhost, so the
    real client IP only exists in ``CF-Connecting-IP``.
    """
    for header in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def client_identifier(request) -> str:
    """Hash of the browser fingerprint header, falling back to the User-Agent.

    Hashed so the raw fingerprint is never stored — we only need equality.
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


class DailyQuota:
    """Fixed-window daily counters keyed by browser fingerprint and by IP.

    A request is refused when *either* counter is exhausted — matching the
    intent "same IP **or** same browser, fewer than N per day".
    """

    def __init__(self, db_path: Path, per_client: int, per_ip: int) -> None:
        self._path = Path(db_path)
        self._per_client = max(0, int(per_client))
        self._per_ip = max(0, int(per_ip))
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
                CREATE TABLE IF NOT EXISTS quota (
                    day   TEXT    NOT NULL,
                    scope TEXT    NOT NULL,
                    key   TEXT    NOT NULL,
                    used  INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, scope, key)
                )
                """
            )

    @staticmethod
    def _used(conn: sqlite3.Connection, day: str, scope: str, key: str) -> int:
        row = conn.execute(
            "SELECT used FROM quota WHERE day = ? AND scope = ? AND key = ?",
            (day, scope, key),
        ).fetchone()
        return int(row[0]) if row else 0

    def _snapshot(self, conn: sqlite3.Connection, day: str, ident: str, ip: str) -> tuple[int, int]:
        return (
            self._used(conn, day, "client", ident),
            self._used(conn, day, "ip", ip),
        )

    def _remaining(self, client_used: int, ip_used: int) -> int:
        return max(0, min(self._per_client - client_used, self._per_ip - ip_used))

    # -- public API --------------------------------------------------------
    @property
    def limits(self) -> dict[str, int]:
        """Configured daily limits."""
        return {"client": self._per_client, "ip": self._per_ip}

    def peek(self, ident: str, ip: str) -> dict:
        """Report usage without consuming anything."""
        day = today_key()
        with self._lock, self._connect() as conn:
            client_used, ip_used = self._snapshot(conn, day, ident, ip)
        return {
            "day": day,
            "clientUsed": client_used,
            "ipUsed": ip_used,
            "clientLimit": self._per_client,
            "ipLimit": self._per_ip,
            "remaining": self._remaining(client_used, ip_used),
        }

    def consume(self, ident: str, ip: str) -> dict:
        """Reserve one unit, or report which counter is exhausted.

        Both counters move together inside a single transaction, so a refused
        request never burns quota.
        """
        day = today_key()
        with self._lock, self._connect() as conn:
            client_used, ip_used = self._snapshot(conn, day, ident, ip)
            if client_used >= self._per_client or ip_used >= self._per_ip:
                return {
                    "allowed": False,
                    "blockedBy": "client" if client_used >= self._per_client else "ip",
                    "day": day,
                    "clientUsed": client_used,
                    "ipUsed": ip_used,
                    "clientLimit": self._per_client,
                    "ipLimit": self._per_ip,
                    "remaining": 0,
                }
            for scope, key in (("client", ident), ("ip", ip)):
                conn.execute(
                    """
                    INSERT INTO quota (day, scope, key, used) VALUES (?, ?, ?, 1)
                    ON CONFLICT (day, scope, key) DO UPDATE SET used = used + 1
                    """,
                    (day, scope, key),
                )
            conn.commit()
            client_used += 1
            ip_used += 1
            return {
                "allowed": True,
                "day": day,
                "clientUsed": client_used,
                "ipUsed": ip_used,
                "clientLimit": self._per_client,
                "ipLimit": self._per_ip,
                "remaining": self._remaining(client_used, ip_used),
            }

    def purge_before(self, day: str) -> int:
        """Delete counters older than ``day``; returns rows removed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM quota WHERE day < ?", (day,))
            conn.commit()
            return int(cur.rowcount or 0)


def quota_headers(info: dict) -> dict[str, str]:
    """Response headers the client uses to show how many shots are left."""
    return {
        "X-Quota-Limit": str(info.get("clientLimit", "")),
        "X-Quota-Remaining": str(info.get("remaining", "")),
        "X-Quota-Day": str(info.get("day", "")),
    }

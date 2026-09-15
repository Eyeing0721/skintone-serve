"""SQLite metadata plus the original-image archive (contract sections 2.5, 2.6, 7).

Two stores, one lifecycle:

* ``{data_dir}/skintone.sqlite3`` -- request metadata, the de-identified result
  JSON, and the archive path when an image was kept.
* ``{data_dir}/archive/{yyyy-mm}/{request_id}.jpg`` -- the original image, with
  EXIF stripped except orientation and timestamp.

The mechanisms are unconditional: ``consent.storeImage=false`` never writes an
image, ``DELETE`` removes metadata *and* file, and startup purges entries older
than the configured retention. *How long* to retain is a policy decision exposed
as :data:`skintone.config.Settings.retention_days`; this module only enforces it.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    request_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    mode         TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    payload      TEXT NOT NULL,
    image_path   TEXT,
    stored       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS results_created_at ON results (created_at);
CREATE TABLE IF NOT EXISTS card_profiles (
    profile_id   TEXT PRIMARY KEY,
    card_id      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    measured     TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StoredResult:
    """A row of the ``results`` table.

    Attributes:
        request_id: Server-generated id, also the archive file stem.
        created_at: ISO-8601 UTC timestamp with a ``Z`` suffix.
        mode: ``nocard`` / ``card`` / ``calibrate``.
        spec_version: Contract version the result was produced under.
        payload: The full de-identified response document.
        image_path: Archive path when the original was kept, else ``None``.
    """

    request_id: str
    created_at: str
    mode: str
    spec_version: str
    payload: dict[str, Any]
    image_path: str | None


def _utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with a ``Z`` suffix and no microseconds."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def strip_exif(image_bytes: bytes) -> bytes:
    """Re-encode an image as JPEG with metadata removed except orientation/time.

    Args:
        image_bytes: The uploaded file bytes (any format Pillow can open).

    Returns:
        JPEG bytes carrying at most the EXIF orientation and capture timestamp.

    Raises:
        ValueError: when the payload is not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            orientation: int | None = None
            timestamp: str | None = None
            try:
                exif = source.getexif()
                orientation = exif.get(config.EXIF_ORIENTATION_TAG)
                timestamp = exif.get(config.EXIF_DATETIME_ORIGINAL_TAG)
                if timestamp is None:
                    timestamp = exif.get_ifd(0x8769).get(config.EXIF_DATETIME_ORIGINAL_TAG)
                if timestamp is None:
                    timestamp = exif.get(config.EXIF_DATETIME_TAG)
            except Exception:  # noqa: BLE001 - malformed EXIF must not fail the request
                orientation, timestamp = None, None

            image = source.convert("RGB") if source.mode not in ("RGB", "L") else source.copy()
            if timestamp is not None and not isinstance(timestamp, str):
                timestamp = str(timestamp)
            kept = Image.Exif()
            if orientation:
                kept[config.EXIF_ORIENTATION_TAG] = orientation
            if timestamp:
                kept[config.EXIF_DATETIME_ORIGINAL_TAG] = timestamp
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=config.ARCHIVE_JPEG_QUALITY,
                exif=kept.tobytes() if len(kept) else b"",
            )
            return buffer.getvalue()
    except Exception as error:  # noqa: BLE001 - surfaced as BAD_IMAGE by the API layer
        raise ValueError(f"cannot decode image: {error}") from error


class Storage:
    """Metadata database and original-image archive.

    All methods are safe to call from multiple threads: each statement uses its
    own short-lived connection guarded by a lock, which is more than enough for a
    single-user local service and avoids cross-thread connection sharing issues.
    """

    def __init__(self, data_dir: Path, retention_days: int) -> None:
        """Create the data directories and schema.

        Args:
            data_dir: Root data directory (``SKINTONE_DATA_DIR``).
            retention_days: Age in days after which archived items are purged.
        """
        self.data_dir = Path(data_dir)
        self.retention_days = int(retention_days)
        self.db_path = self.data_dir / config.DB_FILENAME
        self.archive_dir = self.data_dir / config.ARCHIVE_DIRNAME
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection to the metadata database."""
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def archive_path_for(self, created_at: datetime, request_id: str) -> Path:
        """Return the archive path for a request.

        Args:
            created_at: Capture/processing time, used for the ``{yyyy-mm}`` bucket.
            request_id: Server-generated request id.

        Returns:
            ``{archive_dir}/{yyyy-mm}/{request_id}.jpg``.
        """
        bucket = created_at.astimezone(timezone.utc).strftime(config.ARCHIVE_DATE_FORMAT)
        return self.archive_dir / bucket / f"{request_id}{config.ARCHIVE_SUFFIX}"

    def save_result(
        self,
        request_id: str,
        created_at: datetime,
        mode: str,
        payload: dict[str, Any],
        image_bytes: bytes | None,
    ) -> str | None:
        """Persist a result, optionally archiving the original image.

        Args:
            request_id: Server-generated request id.
            created_at: Processing timestamp (UTC).
            mode: ``nocard`` / ``card`` / ``calibrate``.
            payload: The full de-identified response document.
            image_bytes: Original image bytes, or ``None`` when the user did not
                consent to storage -- in which case nothing is written to disk.

        Returns:
            The archive path as a string when an image was stored, else ``None``.
        """
        stored_path: str | None = None
        if image_bytes is not None:
            stripped = strip_exif(image_bytes)
            path = self.archive_path_for(created_at, request_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(stripped)
            stored_path = str(path)

        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO results "
                "(request_id, created_at, mode, spec_version, payload, image_path, stored) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    _iso(created_at),
                    mode,
                    config.SPEC_VERSION,
                    json.dumps(payload, ensure_ascii=False),
                    stored_path,
                    1 if stored_path else 0,
                ),
            )
        return stored_path

    def get_result(self, request_id: str) -> StoredResult | None:
        """Return a stored result by id, or ``None`` when it does not exist."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM results WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredResult(
            request_id=row["request_id"],
            created_at=row["created_at"],
            mode=row["mode"],
            spec_version=row["spec_version"],
            payload=json.loads(row["payload"]),
            image_path=row["image_path"],
        )

    def delete_result(self, request_id: str) -> bool:
        """Delete a result and its archived original.

        Args:
            request_id: Server-generated request id.

        Returns:
            True when a row existed and was removed (the archive file is removed
            too, whether or not it was still present).
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT image_path FROM results WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM results WHERE request_id = ?", (request_id,))
        if row["image_path"]:
            Path(row["image_path"]).unlink(missing_ok=True)
        return True

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete results (and archived images) older than the retention window.

        Args:
            now: Reference time, defaults to the current UTC time.

        Returns:
            Number of results removed.
        """
        reference = now or _utc_now()
        cutoff = _iso(reference - timedelta(days=self.retention_days))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT request_id, image_path FROM results WHERE created_at < ?", (cutoff,)
            ).fetchall()
            for row in rows:
                connection.execute(
                    "DELETE FROM results WHERE request_id = ?", (row["request_id"],)
                )
                if row["image_path"]:
                    Path(row["image_path"]).unlink(missing_ok=True)
        return len(rows)

    def stats(self) -> dict[str, int]:
        """Return aggregate counters only -- never any user data.

        Returns:
            ``{"totalRequests": int, "storedImages": int, "diskBytes": int}`` where
            ``diskBytes`` is the size of the archive tree in bytes.
        """
        with self._lock, self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            )
            stored = int(
                connection.execute(
                    "SELECT COUNT(*) FROM results WHERE image_path IS NOT NULL"
                ).fetchone()[0]
            )
        disk = 0
        if self.archive_dir.exists():
            for path in self.archive_dir.rglob("*"):
                if path.is_file():
                    disk += path.stat().st_size
        return {"totalRequests": total, "storedImages": stored, "diskBytes": disk}

    def save_profile(
        self,
        profile_id: str,
        card_id: str,
        created_at: datetime,
        measured_srgb: dict[str, list[int]],
    ) -> None:
        """Persist a card self-calibration profile.

        A profile records what *this printed sheet* actually measures under the
        user's own daylight, so a later analysis can use it as the reference
        instead of the ideal nominal values.

        Args:
            profile_id: Client-chosen id, also used by ``meta.cardProfileId``.
            card_id: Card the profile belongs to.
            created_at: Calibration timestamp (UTC).
            measured_srgb: ``{patch_id: [r, g, b]}`` in **encoded** sRGB 0..255.
        """
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO card_profiles "
                "(profile_id, card_id, created_at, measured) VALUES (?, ?, ?, ?)",
                (profile_id, card_id, _iso(created_at), json.dumps(measured_srgb)),
            )

    def get_profile(self, profile_id: str) -> dict[str, list[int]] | None:
        """Return a stored profile's ``{patch_id: [r, g, b]}`` map, or ``None``."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT measured FROM card_profiles WHERE profile_id = ?", (profile_id,)
            ).fetchone()
        if row is None:
            return None
        return {key: list(value) for key, value in json.loads(row["measured"]).items()}

"""Storage behaviour: EXIF stripping, retention purge and delete cascade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from skintone import storage
from tests import synthetic


def _tiny_image() -> np.ndarray:
    """A 40x30 grey BGR image."""
    return np.full((30, 40, 3), 128, dtype=np.uint8)


def test_archive_path_is_bucketed_by_month(tmp_path) -> None:
    """Archived originals live under ``archive/{yyyy-mm}``."""
    store = storage.Storage(tmp_path, retention_days=30)
    path = store.archive_path_for(datetime(2026, 9, 14, tzinfo=timezone.utc), "01ABC")
    assert path.parent.name == "2026-09"
    assert path.name == "01ABC.jpg"
    assert path.parent.parent == store.archive_dir


def test_store_image_false_writes_nothing(tmp_path) -> None:
    """Without consent no file and no archive path may be created."""
    store = storage.Storage(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    stored = store.save_result("01NOROW", now, "nocard", {"hello": "world"}, image_bytes=None)
    assert stored is None
    assert not any(store.archive_dir.rglob("*.jpg"))
    record = store.get_result("01NOROW")
    assert record is not None and record.image_path is None


def test_store_image_true_writes_a_stripped_archive(tmp_path) -> None:
    """With consent the original is archived as JPEG with EXIF stripped."""
    store = storage.Storage(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    raw = synthetic.jpeg_with_exif(_tiny_image(), orientation=6)
    stored = store.save_result("01WITHROW", now, "card", {"ok": True}, image_bytes=raw)
    assert stored is not None
    archived = store.archive_path_for(now, "01WITHROW")
    assert archived.exists()
    with Image.open(archived) as image:
        exif = image.getexif()
        assert exif.get(274) == 6  # orientation survives
        assert exif.get(36867) is not None  # timestamp survives
        assert not exif.get(270)  # the description does not


def test_delete_removes_metadata_and_file(tmp_path) -> None:
    """``DELETE`` must take the archive file with it (contract 9.7)."""
    store = storage.Storage(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    raw = synthetic.encode_jpeg(_tiny_image())
    store.save_result("01DELETEME", now, "nocard", {"ok": True}, image_bytes=raw)
    archived = store.archive_path_for(now, "01DELETEME")
    assert archived.exists()
    assert store.delete_result("01DELETEME") is True
    assert not archived.exists()
    assert store.get_result("01DELETEME") is None


def test_delete_unknown_result_is_false(tmp_path) -> None:
    """Deleting something that does not exist must report it, not crash."""
    store = storage.Storage(tmp_path, retention_days=30)
    assert store.delete_result("01MISSING") is False


def test_purge_removes_only_expired_items(tmp_path) -> None:
    """Retention must delete old rows + files and keep recent ones."""
    store = storage.Storage(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=45)
    raw = synthetic.encode_jpeg(_tiny_image())
    store.save_result("01OLDRESULT", old, "nocard", {"ok": True}, image_bytes=raw)
    store.save_result("01NEWRESULT", now, "nocard", {"ok": True}, image_bytes=raw)
    old_file = store.archive_path_for(old, "01OLDRESULT")
    assert old_file.exists()

    removed = store.purge_expired(now)
    assert removed == 1
    assert store.get_result("01OLDRESULT") is None
    assert store.get_result("01NEWRESULT") is not None
    assert not old_file.exists()


def test_stats_counts_requests_images_and_bytes(tmp_path) -> None:
    """Aggregates must be numeric and never include user data."""
    store = storage.Storage(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    raw = synthetic.encode_jpeg(_tiny_image())
    store.save_result("01A", now, "nocard", {"ok": True}, image_bytes=raw)
    store.save_result("01B", now, "nocard", {"ok": True}, image_bytes=None)
    stats = store.stats()
    assert stats["totalRequests"] == 2
    assert stats["storedImages"] == 1
    assert stats["diskBytes"] > 0


def test_profiles_round_trip(tmp_path) -> None:
    """A card profile must survive save/load unchanged."""
    store = storage.Storage(tmp_path, retention_days=30)
    profile = {"G90": [241, 240, 238], "G70": [198, 197, 195]}
    store.save_profile("my-card-001", "skintone-a4-v1", datetime.now(timezone.utc), profile)
    assert store.get_profile("my-card-001") == profile
    assert store.get_profile("missing") is None


def test_strip_exif_rejects_garbage() -> None:
    """A non-image payload must raise ``ValueError`` (surfaced as BAD_IMAGE)."""
    import pytest

    with pytest.raises(ValueError):
        storage.strip_exif(b"this is not an image")

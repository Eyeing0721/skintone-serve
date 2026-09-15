"""Orchestration helpers: request ids, decoding, gate fusion and level caps."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from skintone import analysis, config, schemas
from tests import synthetic


def test_request_ids_are_26_char_sortable_and_unique() -> None:
    """Ids must be ULID-shaped so they sort by creation time."""
    ids = [analysis.new_request_id() for _ in range(50)]
    assert all(len(value) == 26 for value in ids)
    assert len(set(ids)) == len(ids)
    early = analysis.new_request_id(datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = analysis.new_request_id(datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert early < late


def test_decode_image_applies_the_exif_orientation() -> None:
    """A rotated JPEG must be decoded upright."""
    image = np.full((20, 60, 3), 200, dtype=np.uint8)  # wide
    raw = synthetic.jpeg_with_exif(image, orientation=6)  # 6 = rotate 90 CW
    decoded = analysis.decode_image(raw)
    assert decoded.shape[0] > decoded.shape[1]


def test_decode_image_rejects_garbage() -> None:
    """A non-image must map onto ``BAD_IMAGE``."""
    with pytest.raises(schemas.ApiError) as error:
        analysis.decode_image(b"definitely not an image")
    assert error.value.code == "BAD_IMAGE"
    assert error.value.status_code == 400


def test_decode_image_channel_order_is_bgr() -> None:
    """OpenCV order must be preserved: sRGB white stays white either way."""
    raw = synthetic.encode_png(np.full((10, 10, 3), 77, dtype=np.uint8))
    decoded = analysis.decode_image(raw)
    assert decoded.shape == (10, 10, 3)
    assert int(decoded[0, 0, 0]) == 77


def _measured(**overrides: float) -> dict[str, float]:
    """A full set of gate measurements that all pass comfortably."""
    base = {
        "ccm_delta_e": 1.0,
        "clipping": 0.001,
        "specular": 0.01,
        "roi_area": 20000.0,
        "roi_dispersion": 2.0,
        "illuminant_residual": 0.002,
    }
    base.update(overrides)
    return base


def test_gate_satisfaction_is_monotonic() -> None:
    """Satisfaction must fall as an "at most" gate approaches its threshold."""
    spec = analysis.GateSpec("clipping", 0.02, True, "", "clip")
    values = [0.0, 0.004, 0.008, 0.012, 0.016, 0.02]
    scores = [analysis._gate_satisfaction(spec, value) for value in values]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(0.0)


def test_min_type_gate_caps_at_one() -> None:
    """``roi_area`` must not score above 1.0 just because the ROI is huge."""
    spec = analysis.GateSpec("roi_area", 4000.0, False, "", "area")
    assert analysis._gate_satisfaction(spec, 100000.0) == pytest.approx(1.0)
    assert analysis._gate_satisfaction(spec, 2000.0) == pytest.approx(0.5)


def test_card_mode_reports_every_card_gate() -> None:
    """The card route must expose all five gates with measured values."""
    confidence, warnings = analysis.build_confidence("card", _measured(), [], False)
    assert [gate.id for gate in confidence["gates"]] == [
        "ccm_delta_e",
        "clipping",
        "specular",
        "roi_area",
        "roi_dispersion",
    ]
    assert confidence["level"] == "high"
    assert warnings == []


def test_nocard_caps_confidence_at_medium() -> None:
    """Contract 9.4: no-card may never report ``high``."""
    confidence, warnings = analysis.build_confidence("nocard", _measured(), [], False)
    assert confidence["level"] == "medium"
    assert any("无卡" in text for text in warnings)


def test_failed_ccm_gate_downgrades_to_low() -> None:
    """An untrustworthy calibration must not be reported as high confidence."""
    confidence, warnings = analysis.build_confidence(
        "card", _measured(ccm_delta_e=9.0), [], False
    )
    assert confidence["level"] == "low"
    ccm_gate = next(gate for gate in confidence["gates"] if gate.id == "ccm_delta_e")
    assert ccm_gate.passed is False
    assert "9" in ccm_gate.message
    assert any("色卡标定残差" in text for text in warnings)


def test_failed_roi_area_is_insufficient() -> None:
    """Too few pixels means no conclusion at all, not a low-confidence one."""
    confidence, warnings = analysis.build_confidence("card", _measured(roi_area=10.0), [], False)
    assert confidence["level"] == "insufficient"
    assert any("有效皮肤像素不足" in text for text in warnings)
    assert any("测不准" in text for text in warnings)


def test_failed_illuminant_residual_downgrades_to_low() -> None:
    """Contract 5B: a large projection residual means ``ILLUMINANT_UNRELIABLE``."""
    confidence, warnings = analysis.build_confidence(
        "nocard", _measured(illuminant_residual=0.09), [], False
    )
    assert confidence["level"] == "low"
    assert any("ILLUMINANT_UNRELIABLE" in text for text in warnings)


def test_score_thresholds_match_the_contract() -> None:
    """The level cut points must be exactly the contract's."""
    assert config.CONFIDENCE_HIGH_MIN == 0.75
    assert config.CONFIDENCE_MEDIUM_MIN == 0.50
    assert config.CONFIDENCE_LOW_MIN == 0.25
    assert config.MODE_MAX_CONFIDENCE["nocard"] == "medium"


def test_gate_messages_always_contain_numbers() -> None:
    """Contract 5: every gate must report its measured value in the message."""
    confidence, _ = analysis.build_confidence("card", _measured(), [], False)
    for gate in confidence["gates"]:
        assert any(character.isdigit() for character in gate.message), gate.message


def test_uptime_is_non_negative() -> None:
    """Uptime must never go backwards."""
    import time

    assert analysis.uptime_seconds(time.monotonic()) >= 0.0


def test_api_error_codes_map_to_the_contract_statuses() -> None:
    """Each contract code must carry a sensible HTTP status."""
    assert schemas.ApiError("PAYLOAD_TOO_LARGE", "x").status_code == 413
    assert schemas.ApiError("UNAUTHORIZED", "x").status_code == 401
    assert schemas.ApiError("RATE_LIMITED", "x").status_code == 429
    assert schemas.ApiError("CARD_PROFILE_NOT_FOUND", "x").status_code == 404
    assert schemas.ApiError("SOMETHING_ELSE", "x").code == "INTERNAL"


def test_calibrate_then_analyze_uses_the_profile(tmp_path) -> None:
    """A self-calibrated profile must be usable as the card reference."""
    from fastapi.testclient import TestClient

    import json

    from skintone.main import create_app

    app = create_app(config.Settings(data_dir=tmp_path / "data", rate_limit="1000/minute"))
    card = synthetic.encode_jpeg(synthetic.render_card())
    with TestClient(app) as client:
        meta = {
            "mode": "calibrate",
            "cardId": "skintone-a4-v1",
            "cardProfileId": "my-card-001",
            "capture": {"illuminantGuess": "daylight"},
        }
        response = client.post(
            "/v1/card/skintone-a4-v1/calibrate",
            files={"image": ("card.jpg", card, "image/jpeg")},
            data={"meta": json.dumps(meta)},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["profileId"] == "my-card-001"
        assert body["grayNeutrality"]["passed"] is True
        assert len(body["measuredSrgb"]) >= config.CCM_MIN_PATCHES
        assert body["quality"]["level"] == "good"

    app2 = create_app(config.Settings(data_dir=tmp_path / "data", rate_limit="1000/minute"))
    with TestClient(app2) as client:
        analyze_meta = {
            "mode": "card",
            "cardId": "skintone-a4-v1",
            "cardProfileId": "my-card-001",
            "capture": {"illuminantGuess": "daylight"},
            "consent": {"storeImage": False},
        }
        response = client.post(
            "/v1/analyze",
            files={"image": ("card.jpg", card, "image/jpeg")},
            data={
                "meta": json.dumps(analyze_meta),
                "rois": json.dumps(
                    {"skin": [{"label": "jaw", "points": [[40, 40], [120, 40], [120, 120], [40, 120]]}]}
                ),
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["calibration"] is not None
        assert any("自标定" in text for text in body["warnings"])

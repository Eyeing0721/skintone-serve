"""The two acceptance tests the contract names explicitly (sections 9.3 and 9.4).

9.3 -- render a card that conforms to the specification plus skin blocks with
known CIELAB truth, push the whole scene through a known illuminant, POST it to
``/v1/analyze``, and require the returned ``labD65`` to be within CIEDE2000 3.0 of
the truth. This is the test that makes every other claim in the project checkable.

9.4 -- ``mode=nocard`` on a card-free synthetic scene must never report
``confidence.level="high"``, and every gate must carry a measured value.

The scene is rendered once per session; it is a genuine 6 MP photograph standing
in for a real capture (perspective-warped card, warm illuminant, sensor noise,
JPEG).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pytest

from skintone import config
from skintone.color.ciede2000 import delta_e00
from tests import synthetic

#: Ground-truth CIELAB (D65) of the skin blocks in the card scene.
SKIN_TRUTH = [
    (62.14, 12.42, 18.91),
    (44.80, 15.60, 21.40),
]

#: The simulated capture conditions.
CAPTURE_ILLUMINANT = synthetic.warm_illuminant_xy()


@lru_cache(maxsize=1)
def _card_capture() -> tuple[bytes, list[list[list[float]]], dict]:
    """Render, light, noise and JPEG-encode the card scene (cached)."""
    scene, polygons, info = synthetic.render_scene(SKIN_TRUTH)
    lit = synthetic.simulate_illuminant(scene, CAPTURE_ILLUMINANT, exposure=0.90)
    noisy = synthetic.add_sensor_noise(lit, sigma=1.0)
    return synthetic.encode_jpeg(noisy, quality=92), polygons, info


@lru_cache(maxsize=1)
def _nocard_capture() -> tuple[bytes, list[list[list[float]]], dict]:
    """Render, light, noise and JPEG-encode the card-free scene (cached)."""
    scene, polygons, info = synthetic.render_nocard_scene((58.30, 13.10, 19.70))
    lit = synthetic.simulate_illuminant(scene, CAPTURE_ILLUMINANT, exposure=0.95)
    noisy = synthetic.add_sensor_noise(lit, sigma=1.0, seed=424242)
    return synthetic.encode_jpeg(noisy, quality=92), polygons, info


def _card_meta() -> dict:
    """The ``meta`` object for a card-mode capture."""
    return {
        "mode": "card",
        "cardId": "skintone-a4-v1",
        "capture": {"wbLocked": True, "raw": False, "flash": False, "illuminantGuess": "daylight"},
        "consent": {"storeImage": True, "acceptedAt": "2026-09-14T10:00:00Z"},
        "clientVersion": "0.1.0",
    }


def test_9_3_card_route_recovers_lab_d65_within_delta_e_3(client) -> None:
    """The contract's headline acceptance test: ΔE00 < 3 against known truth."""
    image, polygons, info = _card_capture()
    response = client.post(
        "/v1/analyze",
        files={"image": ("capture.jpg", image, "image/jpeg")},
        data={
            "meta": json.dumps(_card_meta()),
            "rois": json.dumps(
                {"skin": [{"label": "jaw", "points": polygons[0]}]}
            ),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    recovered = body["skin"]["labD65"]
    truth = np.asarray(SKIN_TRUTH[0], dtype=np.float64)
    measured = np.asarray([recovered["L"], recovered["a"], recovered["b"]], dtype=np.float64)
    difference = delta_e00(truth, measured)
    print(
        f"\n[9.3] truth Lab={truth.round(2).tolist()} "
        f"recovered Lab={measured.round(2).tolist()} deltaE00={difference:.3f}"
    )
    assert difference < 3.0, f"deltaE00 {difference:.3f} exceeds the 3.0 gate"

    calibration = body["calibration"]
    assert calibration is not None
    assert calibration["ccmKind"] in {"3x3", "3x4"}
    assert calibration["deltaE00Mean"] < config.CCM_DELTA_E_MAX
    print(
        f"[9.3] ccmKind={calibration['ccmKind']} "
        f"ccm deltaE00 mean={calibration['deltaE00Mean']:.3f} "
        f"max={calibration['deltaE00Max']:.3f} "
        f"patches={len(calibration['perPatch'])}"
    )
    assert len(calibration["perPatch"]) >= config.CCM_MIN_PATCHES

    assert body["confidence"]["level"] == "high"
    assert body["confidence"]["score"] >= config.CONFIDENCE_HIGH_MIN
    assert body["skin"]["depthClass"] == "intermediate"
    assert body["illuminant"]["method"] == "card"
    assert body["illuminant"]["adaptation"] == "Bradford"
    print(
        f"[9.3] confidence={body['confidence']['level']} "
        f"score={body['confidence']['score']} "
        f"cct={body['illuminant']['cct']} duv={body['illuminant']['duv']}"
    )


def test_9_3_second_skin_block_also_matches(client) -> None:
    """A second, darker block must be recovered from the same calibration."""
    image, polygons, _ = _card_capture()
    response = client.post(
        "/v1/analyze",
        files={"image": ("capture.jpg", image, "image/jpeg")},
        data={
            "meta": json.dumps(_card_meta()),
            "rois": json.dumps({"skin": [{"label": "neck", "points": polygons[1]}]}),
        },
    )
    assert response.status_code == 200, response.text
    recovered = response.json()["skin"]["labD65"]
    truth = np.asarray(SKIN_TRUTH[1], dtype=np.float64)
    measured = np.asarray([recovered["L"], recovered["a"], recovered["b"]], dtype=np.float64)
    difference = delta_e00(truth, measured)
    print(f"\n[9.3b] deltaE00={difference:.3f} recovered={measured.round(2).tolist()}")
    assert difference < 3.0


def test_9_4_nocard_never_reports_high_confidence(client) -> None:
    """The no-card route must stay honest about what it cannot prove."""
    image, polygons, _ = _nocard_capture()
    meta = _card_meta()
    meta["mode"] = "nocard"
    meta.pop("cardId")
    response = client.post(
        "/v1/analyze",
        files={"image": ("capture.jpg", image, "image/jpeg")},
        data={
            "meta": json.dumps(meta),
            "rois": json.dumps({"skin": [{"label": "jaw", "points": polygons[0]}]}),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    level = body["confidence"]["level"]
    print(
        f"\n[9.4] level={level} score={body['confidence']['score']} "
        f"illuminant={body['illuminant']}"
    )
    assert level in {"medium", "low", "insufficient"}
    assert level != "high"

    gates = body["confidence"]["gates"]
    gate_ids = [gate["id"] for gate in gates]
    assert gate_ids == ["clipping", "specular", "roi_area", "roi_dispersion", "illuminant_residual"]
    for gate in gates:
        assert isinstance(gate["value"], (int, float)), gate
        assert isinstance(gate["threshold"], (int, float)), gate
        assert gate["message"], gate
        print(
            f"[9.4] gate {gate['id']}: value={gate['value']:.4f} "
            f"threshold={gate['threshold']} passed={gate['passed']}"
        )

    if level == "insufficient":
        assert body["advice"] is None
        assert body["warnings"]
    else:
        assert body["advice"] is not None
        assert body["advice"]["palette"]


def test_nocard_recovers_the_right_depth_class(client) -> None:
    """Without a card the depth axis must still land in the correct ITA band.

    The no-card route has no calibration residual to prove itself with, so this
    test only requires the *class* to match the truth and the colour difference to
    stay in single digits -- it deliberately does not assert the ΔE00 < 3 bound
    that the card route is held to.
    """
    from skintone.color.skin import depth_class

    truth = _nocard_capture()[2]["lab"]
    image, polygons, _ = _nocard_capture()
    meta = _card_meta()
    meta["mode"] = "nocard"
    meta.pop("cardId")
    response = client.post(
        "/v1/analyze",
        files={"image": ("capture.jpg", image, "image/jpeg")},
        data={
            "meta": json.dumps(meta),
            "rois": json.dumps({"skin": [{"label": "jaw", "points": polygons[0]}]}),
        },
    )
    assert response.status_code == 200
    body = response.json()
    recovered = body["skin"]["labD65"]
    measured = np.asarray([recovered["L"], recovered["a"], recovered["b"]], dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    difference = delta_e00(truth_array, measured)
    print(
        f"\n[nocard] truth={truth_array.round(2).tolist()} measured={measured.round(2).tolist()} "
        f"deltaE00={difference:.3f} class={body['skin']['depthClass']}"
    )
    assert body["skin"]["depthClass"] == depth_class(float(truth_array[0]), float(truth_array[2]))
    assert difference < 6.0

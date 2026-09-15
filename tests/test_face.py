"""Haar fallbacks and semantic illuminant candidates (contract section 5B)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from skintone import config
from skintone.vision import face


def _blank(size: int = 320) -> np.ndarray:
    """A uniform mid-grey image."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_no_face_on_a_blank_image() -> None:
    """The cascade must return an empty list rather than a guess."""
    assert face.detect_faces(_blank()) == []


def test_face_mask_covers_the_box_with_padding() -> None:
    """The exclusion mask must be slightly larger than the face box."""
    image = _blank(200)
    mask = face.face_mask(image, [(50, 50, 60, 60)])
    assert mask[80, 80] == 255
    assert mask[10, 10] == 0
    assert int(np.count_nonzero(mask)) > 60 * 60


def test_auto_skin_polygons_are_inside_the_image() -> None:
    """Auto ROIs must be clamped to the frame."""
    image = _blank(200)
    polygons = face.auto_skin_polygons(image, (150, 150, 60, 60))
    assert polygons
    for polygon in polygons:
        for x, y in polygon["points"]:  # type: ignore[union-attr]
            assert 0 <= x <= 199
            assert 0 <= y <= 199


def test_auto_skin_polygons_are_labelled_jaw_and_neck() -> None:
    """The two fallback regions must be labelled as the contract expects."""
    polygons = face.auto_skin_polygons(_blank(400), (100, 100, 200, 200))
    assert [polygon["label"] for polygon in polygons] == ["jaw", "neck"]


def test_sclera_candidates_require_a_detected_eye() -> None:
    """Sclera sampling is gated on Haar actually finding an eye.

    A painted rectangle is not an eye, so the correct behaviour is "no
    candidates" -- the caller then falls back to teeth/background rather than
    inventing a white reference.
    """
    image = _blank(240)
    image[60:180, 50:170] = (200, 190, 180)  # BGR: a bright, warm "sclera"
    assert face.sclera_candidates(image, [(50, 60, 120, 120)]).shape[0] == 0


def test_bright_low_saturation_selects_the_achromatic_pixels() -> None:
    """The core sclera/teeth selector must keep the least saturated pixels."""
    image = _blank(120)
    image[:, :] = (60, 60, 60)  # too dark: fails MIN_VALUE
    image[0:40, :] = (180, 190, 200)  # BGR -> RGB(200,190,180): bright, S ~ 0.10
    image[40:80, :] = (40, 40, 240)  # bright but saturated (S ~ 0.83)
    pixels = face._bright_low_saturation(image, (0, 0, 120, 120), 0.18, 90, 5000)
    assert pixels.shape[0] == 40 * 120
    # Linear RGB of the warm patch: red above blue, all well above zero.
    median = np.median(pixels, axis=0)
    assert median[0] > median[2]
    assert float(np.mean(median)) > 0.3


def test_sclera_candidates_empty_when_no_eye() -> None:
    """A flat image has no sclera, and that must be reported as "no candidate"."""
    assert face.sclera_candidates(_blank(), [(50, 50, 100, 100)]).shape[0] == 0


def test_teeth_candidates_find_a_bright_mouth() -> None:
    """A bright, desaturated mouth region must yield candidates."""
    image = _blank(240)
    image[150:170, 95:145] = (235, 235, 230)  # teeth, BGR
    candidates = face.teeth_candidates(image, [(60, 60, 120, 140)])
    assert candidates.shape[0] > 0
    assert float(np.mean(np.median(candidates, axis=0))) > 0.5


def test_teeth_candidates_empty_without_a_mouth() -> None:
    """Nothing bright in the mouth band means no candidates."""
    image = np.full((240, 240, 3), 20, dtype=np.uint8)
    assert face.teeth_candidates(image, [(60, 60, 120, 140)]).shape[0] == 0


def test_cascades_are_loaded_from_the_opencv_package() -> None:
    """The Haar XMLs must come from ``cv2.data.haarcascades`` with zero download."""
    cascade = face._cascade(config.HAAR_FACE_CASCADE)
    assert not cascade.empty()
    eye_cascade = face._cascade(config.HAAR_EYE_CASCADE)
    assert not eye_cascade.empty()


def test_detect_eyes_on_a_flat_face_box_is_empty() -> None:
    """A featureless box must yield no eyes."""
    assert face.detect_eyes(_blank(), (10, 10, 200, 200)) == []


def test_bright_low_saturation_respects_the_pixel_limit() -> None:
    """The selection helper must never return more than its limit."""
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    pixels = face._bright_low_saturation(image, (0, 0, 100, 100), 0.5, 100, 37)
    assert pixels.shape[0] == 37


def test_bright_low_saturation_on_an_empty_box() -> None:
    """A degenerate box must return an empty array, not raise."""
    pixels = face._bright_low_saturation(_blank(), (10, 10, 0, 0), 0.5, 10, 10)
    assert pixels.shape == (0, 3)


@pytest.mark.parametrize("cascade_name", [config.HAAR_SMILE_CASCADE])
def test_smile_cascade_exists(cascade_name: str) -> None:
    """The optional smile cascade must ship with OpenCV too."""
    assert cv2.data.haarcascades
    assert (face._cascade(cascade_name) is not None)

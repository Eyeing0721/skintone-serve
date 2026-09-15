"""Card detection, sampling and the least-squares CCM (contract section 5A)."""

from __future__ import annotations

import numpy as np
import pytest

from skintone import config
from skintone.cards import definitions
from skintone.color import srgb
from skintone.vision import card as card_vision
from tests import synthetic

RENDER_CACHE: dict[int, np.ndarray] = {}


def rendered_card() -> np.ndarray:
    """Render the canonical card once per session (it is expensive)."""
    key = id(np)
    if key not in RENDER_CACHE:
        RENDER_CACHE[key] = synthetic.render_card()
    return RENDER_CACHE[key]


def test_all_four_markers_are_detected() -> None:
    """A rendered card must yield ArUco ids 0-3."""
    markers = card_vision.detect_markers(rendered_card())
    assert set(markers) == {0, 1, 2, 3}
    for corners in markers.values():
        assert corners.shape == (4, 2)


def test_homography_round_trips_a_rendered_card() -> None:
    """Warping the canonical card with its own homography must be near identity."""
    card = rendered_card()
    markers = card_vision.detect_markers(card)
    homography = card_vision.homography_to_canonical(markers)
    assert homography is not None
    warped = card_vision.warp_to_canonical(card, homography)
    assert warped.shape == (config.CARD_HEIGHT_PX, config.CARD_WIDTH_PX, 3)
    assert np.mean(np.abs(warped.astype(np.int16) - card.astype(np.int16))) < 1.0


def test_homography_requires_all_four_markers() -> None:
    """Three markers is not enough -- the contract says four."""
    markers = card_vision.detect_markers(rendered_card())
    markers.pop(2)
    assert card_vision.homography_to_canonical(markers) is None


def test_sampled_patches_equal_their_nominal_values() -> None:
    """Sampling a clean render must reproduce the nominal sRGB values exactly.

    The pure-white patch is intentionally unusable: any channel at or above
    ``config.CCM_DROP_SATURATED_SRGB8`` carries no recoverable colour.
    """
    samples = {sample.id: sample for sample in card_vision.sample_all_patches(rendered_card())}
    for patch in definitions.PATCHES:
        sample = samples[patch.id]
        if max(patch.nominal_srgb) >= config.CCM_DROP_SATURATED_SRGB8:
            assert not sample.usable, patch.id
            continue
        assert sample.usable, patch.id
        assert np.allclose(
            sample.measured_srgb8, np.asarray(patch.nominal_srgb, dtype=float), atol=1.0
        )


def test_ccm_on_ideal_data_is_the_identity() -> None:
    """With no camera error the least-squares CCM must be (essentially) identity."""
    samples = card_vision.sample_all_patches(rendered_card())
    result = card_vision.choose_ccm(samples)
    assert result.kind == "3x3"
    assert result.delta_e_mean < 0.05
    assert np.allclose(result.matrix, srgb.XYZ_FROM_LINEAR_RGB, atol=0.02)


def test_ccm_recovers_a_known_linear_distortion() -> None:
    """A synthetic camera matrix must be inverted to within a small residual."""
    rng = np.random.default_rng(11)
    camera = np.eye(3) + rng.normal(0.0, 0.05, (3, 3))
    nominal_linear = np.stack(
        [
            srgb.srgb8_to_linear(np.asarray(patch.nominal_srgb, dtype=float))
            for patch in definitions.PATCHES
        ]
    )
    reference_xyz = nominal_linear @ srgb.XYZ_FROM_LINEAR_RGB.T
    measured = nominal_linear @ np.linalg.inv(camera).T
    matrix = card_vision.solve_ccm(measured, reference_xyz, "3x3")
    corrected_lab = srgb.xyz_to_lab(card_vision.apply_ccm(measured, matrix))
    reference_lab = srgb.xyz_to_lab(reference_xyz)
    from skintone.color.ciede2000 import ciede2000

    assert float(np.max(ciede2000(reference_lab, corrected_lab))) < 0.5


def test_ccm_three_by_four_has_four_columns() -> None:
    """The offset model must return a 3x4 matrix."""
    samples = card_vision.sample_all_patches(rendered_card())
    measured = np.stack([s.measured_linear for s in samples])
    reference = np.stack([s.reference_xyz for s in samples])
    matrix = card_vision.solve_ccm(measured, reference, "3x4")
    assert matrix.shape == (3, 4)


def test_ccm_rejects_an_unknown_kind() -> None:
    """An unknown model must raise rather than silently pick one."""
    measured = np.zeros((10, 3))
    reference = np.zeros((10, 3))
    with pytest.raises(ValueError):
        card_vision.solve_ccm(measured, reference, "3x5")


def test_ccm_rejects_mismatched_shapes() -> None:
    """Reference and measured sets must correspond one-to-one."""
    with pytest.raises(ValueError):
        card_vision.solve_ccm(np.zeros((5, 3)), np.zeros((4, 3)), "3x3")


def test_choose_ccm_requires_enough_patches() -> None:
    """Too few usable patches must raise, never produce a bogus matrix."""
    samples = card_vision.sample_all_patches(rendered_card())
    crippled = [sample for sample in samples[:5]]
    with pytest.raises(ValueError):
        card_vision.choose_ccm(crippled)


def test_saturated_patches_are_dropped() -> None:
    """A blown-out patch must be marked unusable and excluded from the fit."""
    card = rendered_card().copy()
    patch = definitions.get_patch("G50")
    assert patch is not None
    x0, y0, x1, y1 = definitions.patch_sample_bounds_px(patch)
    card[y0:y1, x0:x1] = 255
    samples = card_vision.sample_all_patches(card)
    assert not next(sample for sample in samples if sample.id == "G50").usable


def test_offset_model_selected_when_black_is_lifted() -> None:
    """A lifted black level must switch the fit to the 3x4 offset model."""
    card = rendered_card().astype(np.float64)
    card = np.clip(card * 0.80 + 32.0, 0, 255).astype(np.uint8)
    samples = card_vision.sample_all_patches(card)
    result = card_vision.choose_ccm(samples)
    assert result.kind == "3x4"
    assert result.matrix.shape == (3, 4)


def test_card_white_is_neutral_on_a_clean_render() -> None:
    """With no illuminant shift the card's neutrals must read as D65."""
    samples = card_vision.sample_all_patches(rendered_card())
    xy = card_vision.card_white_xy(samples)
    assert xy is not None
    assert float(xy[0]) == pytest.approx(0.3127, abs=0.005)
    assert float(xy[1]) == pytest.approx(0.3290, abs=0.005)


def test_card_white_tracks_a_simulated_illuminant() -> None:
    """Under a warm illuminant the card's neutrals must show that warm cast."""
    warm_xy = synthetic.warm_illuminant_xy()
    shifted = synthetic.simulate_illuminant(rendered_card(), warm_xy, exposure=1.0)
    samples = card_vision.sample_all_patches(shifted)
    xy = card_vision.card_white_xy(samples)
    assert xy is not None
    assert float(xy[0]) > 0.34

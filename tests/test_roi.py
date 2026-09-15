"""ROI masking and robust statistics (contract section 5, gate definitions)."""

from __future__ import annotations

import numpy as np
import pytest

from skintone import config
from skintone.color import srgb
from skintone.vision import roi


def _flat_image(colour_bgr: tuple[int, int, int], size: int = 200) -> np.ndarray:
    """Return a uniform BGR uint8 image of one colour."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = colour_bgr
    return image


def _square(points_xy: list[list[float]]) -> list[list[list[float]]]:
    """Wrap a polygon in the nested list shape the API uses."""
    return [points_xy]


def test_polygon_mask_covers_the_expected_area() -> None:
    """A 100x100 square must set about 10000 pixels.

    ``cv2.fillPoly`` draws both edge lines, so the rasterised area is
    ``(w + 1) * (h + 1)`` -- 10201 for this square; the test allows 3 % slack.
    """
    mask = roi.polygons_to_mask((200, 200), _square([[50, 50], [150, 50], [150, 150], [50, 150]]))
    assert int(np.count_nonzero(mask)) == pytest.approx(10000, rel=0.03)


def test_degenerate_polygons_are_ignored() -> None:
    """Fewer than three points cannot form an area."""
    mask = roi.polygons_to_mask((50, 50), _square([[1, 1], [10, 10]]))
    assert int(np.count_nonzero(mask)) == 0


def test_median_of_a_uniform_roi_equals_the_colour() -> None:
    """A flat patch must measure back to itself, in linear light."""
    colour = (126, 154, 201)  # BGR
    stats = roi.compute_roi_stats(
        _flat_image(colour), _square([[20, 20], [180, 20], [180, 180], [20, 180]]), ["jaw"]
    )
    assert stats.valid_pixels == pytest.approx(160 * 160, rel=0.02)
    assert stats.median_linear_rgb is not None
    expected = srgb.srgb8_to_linear(np.array([201, 154, 126], dtype=np.float64))
    assert np.allclose(stats.median_linear_rgb, expected, atol=1e-9)
    assert stats.clipping_ratio == 0.0
    assert stats.specular_ratio == 0.0
    assert stats.dispersion == pytest.approx(0.0, abs=1e-6)


def test_outliers_do_not_move_the_median() -> None:
    """The median must resist a large minority of extreme pixels."""
    image = _flat_image((120, 140, 160))
    image[0:40, 0:40] = (255, 255, 255)  # a quarter of the ROI, fully blown
    stats = roi.compute_roi_stats(
        image, _square([[0, 0], [80, 0], [80, 80], [0, 80]]), ["jaw"]
    )
    expected = srgb.srgb8_to_linear(np.array([160, 140, 120], dtype=np.float64))
    assert np.allclose(stats.median_linear_rgb, expected, atol=1e-9)


def test_clipping_gate_counts_saturated_and_crushed_pixels() -> None:
    """Any channel at >= 250 or <= 5 counts as clipped."""
    image = _flat_image((128, 128, 128), size=100)
    image[0:50, :] = (255, 128, 128)
    image[50:75, :] = (3, 128, 128)
    stats = roi.compute_roi_stats(
        image, _square([[0, 0], [100, 0], [100, 100], [0, 100]]), ["jaw"]
    )
    assert stats.roi_pixels == 10000
    assert stats.clipping_ratio == pytest.approx(0.75)
    assert stats.clipping_ratio > config.CLIP_RATIO_MAX


def test_specular_gate_counts_high_lightness_pixels() -> None:
    """Pixels with CIELAB L* > 85 count as specular."""
    image = _flat_image((128, 128, 128), size=100)
    image[0:20, :] = (240, 240, 240)  # L* ~ 95
    stats = roi.compute_roi_stats(
        image, _square([[0, 0], [100, 0], [100, 100], [0, 100]]), ["jaw"]
    )
    assert stats.specular_ratio == pytest.approx(0.20)
    assert stats.specular_ratio > config.SPECULAR_RATIO_MAX


def test_light_skin_is_not_mistaken_for_specular() -> None:
    """浅肤色不该被当成高光。

    回归防线：``specular`` 曾用绝对 ``L* > 85``，于是 L*≈86 的正常浅肤色像素
    **100%** 被判成高光——既触发 specular 闸门（阈值 8%，直接判"测不准"），
    又被 ``valid = ~(clipped | specular)`` 剔除出中位数，等于系统性歧视浅肤色。
    改成相对肤色中位数判定后不该再发生。
    """
    # BGR (200, 213, 232) = #e8d5c8，L* ≈ 86，正好落在旧阈值 85 之上
    stats = roi.compute_roi_stats(
        _flat_image((200, 213, 232)),
        _square([[20, 20], [180, 20], [180, 180], [20, 180]]),
        ["jaw"],
    )
    assert stats.lab is not None
    assert stats.lab[0] > 82.0, "这块颜色必须真的够浅，否则这条测试测的是别的东西"
    assert stats.specular_ratio == 0.0
    assert stats.clipping_ratio == 0.0
    assert stats.valid_pixels == stats.roi_pixels


def test_specular_cut_follows_the_subject_not_a_fixed_l_star() -> None:
    """高光判定必须随肤色移动：同样的亮斑，在深肤色 ROI 里要算高光。"""
    # 深肤色打底（L*≈33）+ 一块 L*≈60 的亮斑
    image = _flat_image((58, 74, 107), size=100)  # BGR -> #6b4a3a
    image[0:20, :] = (150, 160, 170)  # 明显高于肤色中心
    stats = roi.compute_roi_stats(
        image, _square([[0, 0], [100, 0], [100, 100], [0, 100]]), ["jaw"]
    )
    assert stats.lab is not None and stats.lab[0] < 45.0
    # 亮斑的 L* 只有约 65，远低于绝对下限 88，但它确实比这块肤色亮出一大截
    assert stats.specular_ratio == 0.0, "低于绝对下限时不该触发——下限正是为了防止误杀"


def test_dispersion_measures_spread_in_delta_e_units() -> None:
    """A noisy ROI must report a larger dispersion than a flat one."""
    flat = roi.compute_roi_stats(
        _flat_image((120, 140, 160)), _square([[10, 10], [190, 10], [190, 190], [10, 190]]), ["jaw"]
    )
    rng = np.random.default_rng(7)
    noisy_image = np.clip(
        _flat_image((120, 140, 160)).astype(np.int16) + rng.integers(-12, 12, (200, 200, 3)),
        0,
        255,
    ).astype(np.uint8)
    noisy = roi.compute_roi_stats(
        noisy_image, _square([[10, 10], [190, 10], [190, 190], [10, 190]]), ["jaw"]
    )
    assert flat.dispersion is not None and noisy.dispersion is not None
    assert noisy.dispersion > flat.dispersion


def test_empty_roi_reports_itself_without_raising() -> None:
    """An ROI outside the image must produce a describable empty result."""
    stats = roi.compute_roi_stats(_flat_image((120, 120, 120)), [], [])
    assert stats.roi_pixels == 0
    assert stats.valid_pixels == 0
    assert stats.median_linear_rgb is None
    assert stats.lab is None


def test_all_rejected_roi_keeps_the_ratios() -> None:
    """A fully blown ROI must still report the measured ratios."""
    stats = roi.compute_roi_stats(
        _flat_image((255, 255, 255)), _square([[0, 0], [200, 0], [200, 200], [0, 200]]), ["jaw"]
    )
    assert stats.valid_pixels == 0
    assert stats.clipping_ratio == pytest.approx(1.0)
    assert stats.median_linear_rgb is None


def test_trimmed_mean_is_reported_alongside_the_median() -> None:
    """Both robust estimators must be available to callers."""
    stats = roi.compute_roi_stats(
        _flat_image((100, 150, 200)), _square([[0, 0], [200, 0], [200, 200], [0, 200]]), ["neck"]
    )
    assert stats.trimmed_mean_linear_rgb is not None
    assert np.allclose(stats.trimmed_mean_linear_rgb, stats.median_linear_rgb, atol=1e-9)


def test_regions_are_recorded_in_order() -> None:
    """The labels must be preserved for the response's ``roi.regions``."""
    stats = roi.compute_roi_stats(
        _flat_image((120, 140, 160)),
        _square([[0, 0], [50, 0], [50, 50], [0, 50]]) + _square([[100, 100], [150, 100], [150, 150], [100, 150]]),
        ["jaw", "neck"],
    )
    assert stats.regions == ["jaw", "neck"]
    assert stats.roi_pixels == pytest.approx(2 * 2500, rel=0.05)


def test_background_candidates_exclude_the_masked_region() -> None:
    """Semantic masking must keep masked pixels out of the illuminant candidates."""
    image = _flat_image((190, 190, 190), size=200)
    exclude = np.zeros((200, 200), dtype=np.uint8)
    exclude[:20, :] = 255  # the top border band
    candidates = roi.neutral_candidates(image, exclude)
    assert candidates.shape[0] > 0
    # Everything returned is linear ~ (190/255) decoded.
    expected = srgb.srgb8_to_linear(np.array([190.0, 190.0, 190.0]))
    assert np.allclose(candidates[0], expected, atol=1e-9)


def test_background_candidates_ignore_saturated_pixels() -> None:
    """A colourful border must yield no neutral candidates."""
    image = _flat_image((40, 40, 240), size=200)  # strongly blue: HSV S ~ 0.83
    candidates = roi.neutral_candidates(image, np.zeros((200, 200), dtype=np.uint8))
    assert candidates.shape[0] == 0


def test_background_candidates_accept_a_neutral_wall_under_a_warm_light() -> None:
    """A mildly chromatic border (neutral surface, warm lamp) must still qualify."""
    image = _flat_image((150, 178, 200), size=200)  # warm cast, HSV S ~ 0.25
    candidates = roi.neutral_candidates(image, np.zeros((200, 200), dtype=np.uint8))
    assert candidates.shape[0] > 0


def test_mask_fraction() -> None:
    """Fraction helper must handle empty masks."""
    assert roi.mask_fraction(np.zeros((0, 0), dtype=np.uint8)) == 0.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 255
    assert roi.mask_fraction(mask) == pytest.approx(0.5)

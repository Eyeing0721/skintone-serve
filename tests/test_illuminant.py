"""Unit tests for chromatic adaptation and illuminant geometry (contract 4 and 5B)."""

from __future__ import annotations

import numpy as np
import pytest

from skintone import config
from skintone.color import adaptation, illuminant
from skintone.color.srgb import D65_XYZ, white_xyz_from_xy


def test_bradford_of_d65_to_d65_is_identity() -> None:
    """Adapting from a white point to itself must be the identity matrix.

    The published Bradford inverse is quoted to seven digits, so the identity is
    recovered to about 1e-7, not to machine epsilon.
    """
    matrix = adaptation.adaptation_matrix(D65_XYZ, D65_XYZ, "Bradford")
    assert np.allclose(matrix, np.eye(3), atol=1e-6)


def test_adaptation_maps_source_white_to_destination_white() -> None:
    """The defining property: white_src must map exactly onto white_dst."""
    source = white_xyz_from_xy((0.4476, 0.4074))  # illuminant A
    matrix = adaptation.adaptation_matrix(source, D65_XYZ, "Bradford")
    assert np.allclose(matrix @ source, D65_XYZ, atol=1e-6)


@pytest.mark.parametrize("method", ["Bradford", "von Kries", "XYZ scaling"])
def test_adaptation_round_trip(method: str) -> None:
    """Going to another white and back must be lossless."""
    source = white_xyz_from_xy((0.3457, 0.3587))  # D50-ish
    there = adaptation.adapt_xyz(D65_XYZ, source, D65_XYZ, method)
    back = adaptation.adapt_xyz(there, D65_XYZ, source, method)
    assert np.allclose(back, D65_XYZ, atol=1e-9)


def test_unknown_method_is_rejected() -> None:
    """An unknown CAT must raise, never silently fall back."""
    with pytest.raises(ValueError):
        adaptation.adaptation_matrix(D65_XYZ, D65_XYZ, "Cat02-ish")


def test_linear_rgb_adaptation_round_trip() -> None:
    """Adapting linear sRGB out and back must be lossless."""
    source = white_xyz_from_xy((0.4476, 0.4074))
    colour = np.array([0.34, 0.24, 0.18])
    there = adaptation.adapt_linear_rgb(colour, source, D65_XYZ)
    back = adaptation.adapt_linear_rgb(there, D65_XYZ, source)
    assert np.allclose(back, colour, atol=1e-9)


def test_planckian_locus_matches_published_chromaticities() -> None:
    """Planckian xy at 2856 K and 6504 K must match the published values."""
    warm = illuminant.planck_xy(2856.0)
    assert warm[0] == pytest.approx(0.4476, abs=0.002)
    assert warm[1] == pytest.approx(0.4074, abs=0.002)
    neutral = illuminant.planck_xy(6504.0)
    assert neutral[0] == pytest.approx(0.3135, abs=0.002)
    assert neutral[1] == pytest.approx(0.3237, abs=0.002)


def test_daylight_locus_reproduces_d65() -> None:
    """The CIE daylight closed form at 6504 K must reproduce the D65 point."""
    xy = illuminant.daylight_xy(6504.0)
    assert xy[0] == pytest.approx(0.31271, abs=1e-4)
    assert xy[1] == pytest.approx(0.32902, abs=1e-4)


@pytest.mark.parametrize("temperature", [2856.0, 4000.0, 5000.0, 6504.0, 10000.0])
def test_planckian_projection_is_self_consistent(temperature: float) -> None:
    """A point taken from a locus must project back to the same CCT with Duv 0."""
    xy = illuminant.planck_xy(temperature)
    cct, duv = illuminant.project_to_locus(xy, "planckian")
    assert cct == pytest.approx(temperature, rel=1e-3)
    assert abs(duv) < 1e-4


def test_duv_sign_follows_the_side_of_the_locus() -> None:
    """Duv must be positive above the locus (greenish) and negative below."""
    base = illuminant.planck_xy(5000.0)
    above = np.array([base[0], base[1] + 0.004])
    below = np.array([base[0], base[1] - 0.004])
    _, duv_above = illuminant.project_to_locus(above, "planckian")
    _, duv_below = illuminant.project_to_locus(below, "planckian")
    assert duv_above > 0.0
    assert duv_below < 0.0


def test_mccamy_approximates_cct() -> None:
    """McCamy is only a seed, but must stay within a few percent of the truth."""
    for temperature in (2856.0, 5000.0, 6504.0):
        xy = illuminant.planck_xy(temperature)
        assert illuminant.cct_mccamy(float(xy[0]), float(xy[1])) == pytest.approx(
            temperature, rel=0.05
        )


def test_uv_round_trip() -> None:
    """xy -> uv -> xy must be lossless."""
    xy = np.array([0.3457, 0.3587])
    assert np.allclose(illuminant.uv_to_xy(illuminant.xy_to_uv(xy)), xy, atol=1e-12)


def test_solve_illuminant_blends_prior_in_mired() -> None:
    """The prior must move the estimate towards the nominal CCT for that guess."""
    xy = illuminant.planck_xy(6000.0)
    without = illuminant.solve_illuminant(xy, "test", "unknown")
    with_prior = illuminant.solve_illuminant(xy, "test", "tungsten")
    assert without["cct"] == pytest.approx(6000.0, rel=1e-3)
    assert with_prior["cct"] < without["cct"]
    assert with_prior["assumedD65"] is False


def test_solve_illuminant_can_assume_d65() -> None:
    """With no measurement at all the result must declare itself as assumed."""
    solved = illuminant.solve_illuminant(None, "d65-assumed", assumed_d65=True)
    assert solved["assumedD65"] is True
    assert solved["cct"] == pytest.approx(6504.0, abs=1.0)


@pytest.mark.parametrize(
    ("guess", "direction"),
    [("tungsten", "warmer"), ("fluorescent", "warmer"), ("shade", "cooler")],
)
def test_d65_fallback_honours_a_known_guess(guess: str, direction: str) -> None:
    """采不到任何中性参考面时，用户选的光源必须真的改变结果。

    回归防线：这里原本在调用处硬写了 ``prior_weight=0.0``，而且
    ``solve_illuminant`` 在无测量时直接提前 return 了 D65，于是
    ``ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK`` 成了永不生效的死配置——
    白炽灯下拍的照片会被按日光处理，而那恰恰是先验最该发挥作用的场景。
    """
    solved = illuminant.solve_illuminant(None, "d65-assumed", guess, assumed_d65=True)
    assert solved["assumedD65"] is True
    assert solved["priorApplied"] is True
    assert solved["priorWeight"] == pytest.approx(
        config.ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK
    )
    if direction == "warmer":
        assert solved["cct"] < illuminant.D65_CCT_K
    else:
        assert solved["cct"] > illuminant.D65_CCT_K


def test_d65_fallback_ignores_an_unknown_guess() -> None:
    """选"不确定"时不该凭空偏向任何色温。"""
    solved = illuminant.solve_illuminant(None, "d65-assumed", "unknown", assumed_d65=True)
    assert solved["cct"] == pytest.approx(illuminant.D65_CCT_K)
    assert solved["priorApplied"] is False
    assert solved["priorWeight"] == 0.0


def test_both_paths_report_whether_the_prior_was_applied() -> None:
    """先验有没有生效必须可查——否则界面上那个下拉框就是个黑盒。"""
    xy = illuminant.daylight_xy(5000.0)
    plain = illuminant.solve_illuminant(xy, "test", "unknown")
    assert plain["priorApplied"] is False
    assert plain["priorWeight"] == 0.0

    nudged = illuminant.solve_illuminant(xy, "test", "tungsten")
    assert nudged["priorApplied"] is True
    assert nudged["priorWeight"] == pytest.approx(config.ILLUMINANT_PRIOR_WEIGHT)
    # 白炽灯先验应当把色温往暖里拉
    assert nudged["cct"] < plain["cct"]


def test_screen_guess_is_the_selfie_scene() -> None:
    """「屏幕光 + 室内灯」是自拍的现实场景，应当把色温往 4800K 一带拉。

    屏幕背光是冷白 LED（约 6500K），室内灯约 4000K，混合后取 4800K。
    它是前端的默认项——最该猜对的情况不该丢给算法。
    """
    assert config.ILLUMINANT_GUESS_CCT["screen"] == pytest.approx(4800.0)
    assert config.ILLUMINANT_GUESS_LOCUS["screen"] == "planckian"

    xy = illuminant.planck_xy(6500.0)
    plain = illuminant.solve_illuminant(xy, "test", "unknown")
    screen = illuminant.solve_illuminant(xy, "test", "screen")
    assert screen["priorApplied"] is True
    assert screen["locus"] == "planckian"
    # 比实测的 6500K 更暖，但不会跑到暖黄灯那一档
    assert screen["cct"] < plain["cct"]
    assert screen["cct"] > config.ILLUMINANT_GUESS_CCT["tungsten"]


def test_solve_illuminant_requires_a_measurement() -> None:
    """Asking for a measured illuminant without one must raise."""
    with pytest.raises(ValueError):
        illuminant.solve_illuminant(None, "test", assumed_d65=False)

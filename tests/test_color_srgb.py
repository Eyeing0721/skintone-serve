"""Unit tests for the hand-written sRGB / CIELAB conversions (contract section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from skintone.color import srgb


def test_d65_chromaticity() -> None:
    """The sRGB white point must land on the D65 chromaticity."""
    xy = srgb.xyz_to_xy(srgb.D65_XYZ)
    assert xy[0] == pytest.approx(0.31271, abs=1e-4)
    assert xy[1] == pytest.approx(0.32902, abs=1e-4)


def test_transfer_function_matches_iec_61966_2_1_at_the_knee() -> None:
    """Both branches must agree at the 0.04045 knee."""
    assert srgb.srgb_to_linear(0.04045) == pytest.approx(0.04045 / 12.92, rel=1e-6)
    assert srgb.linear_to_srgb(0.0031308) == pytest.approx(0.0031308 * 12.92, rel=1e-6)


def test_transfer_function_round_trip() -> None:
    """Decoding then encoding must return the original values."""
    values = np.linspace(0.0, 1.0, 257)
    assert np.allclose(srgb.linear_to_srgb(srgb.srgb_to_linear(values)), values, atol=1e-12)


def test_white_is_lab_100_and_black_is_zero() -> None:
    """sRGB white is CIELAB (100, 0, 0) and black is (0, 0, 0)."""
    assert np.allclose(srgb.srgb8_to_lab([255, 255, 255]), [100.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(srgb.srgb8_to_lab([0, 0, 0]), [0.0, 0.0, 0.0], atol=1e-9)


def test_mid_gray_lightness() -> None:
    """8-bit 128 must give the textbook L* of about 53.585."""
    lightness = float(srgb.srgb8_to_lab([128, 128, 128])[0])
    assert lightness == pytest.approx(53.585, abs=0.01)


def test_lab_round_trip_on_a_saturated_colour() -> None:
    """Lab -> XYZ -> Lab must be an identity to numerical precision."""
    lab = np.array([62.14, 12.42, 18.91])
    assert np.allclose(srgb.xyz_to_lab(srgb.lab_to_xyz(lab)), lab, atol=1e-10)


def test_lch_round_trip() -> None:
    """Lab -> LCh -> Lab must be an identity, with h in [0, 360)."""
    lab = np.array([[62.14, 12.42, 18.91], [30.0, -20.0, -5.0]])
    lch = srgb.lab_to_lch(lab)
    assert np.all((lch[:, 2] >= 0.0) & (lch[:, 2] < 360.0))
    assert np.allclose(srgb.lch_to_lab(lch), lab, atol=1e-12)


def test_hex_round_trip() -> None:
    """A hex colour must survive parse -> linear -> hex."""
    rgb8 = srgb.hex_to_srgb8("#c99a7e")
    assert list(rgb8) == [201, 154, 126]
    assert srgb.linear_to_hex(srgb.srgb8_to_linear(rgb8)) == "#c99a7e"


def test_hex_parser_rejects_bad_input() -> None:
    """A malformed hex string must raise rather than silently misparse."""
    with pytest.raises(ValueError):
        srgb.hex_to_srgb8("#abc")


def test_averaging_linear_is_not_averaging_encoded() -> None:
    """Guard the classic bug: the two are materially different."""
    encoded_mean = float(np.mean([0.0, 1.0]))
    linear_mean = float(np.mean(srgb.srgb_to_linear([0.0, 1.0])))
    assert srgb.linear_to_srgb(linear_mean) == pytest.approx(0.7354, abs=1e-3)
    assert abs(srgb.linear_to_srgb(linear_mean) - encoded_mean) > 0.2

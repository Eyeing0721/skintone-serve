"""Unit tests for skin descriptors: ITA, undertone, chromophore indices."""

from __future__ import annotations

import math

import numpy as np
import pytest

from skintone import config
from skintone.color import skin


def test_ita_matches_the_contract_example() -> None:
    """The contract's worked example must reproduce exactly."""
    assert skin.ita_degrees(62.14, 18.91) == pytest.approx(32.70, abs=0.02)
    assert skin.depth_class(62.14, 18.91) == "intermediate"


@pytest.mark.parametrize(
    ("ita", "expected"),
    [
        (70.0, "very-light"),
        (48.0, "light"),
        (35.0, "intermediate"),
        (20.0, "tan"),
        (0.0, "brown"),
        (-45.0, "dark"),
    ],
)
def test_depth_classes(ita: float, expected: str) -> None:
    """ITA bands must map onto the contract's depth enum."""
    assert skin.depth_class(_l_for_ita(ita), 10.0) == expected


def _l_for_ita(ita: float, b_star: float = 10.0) -> float:
    """Return the ``L*`` that yields a target ITA for a fixed ``b*``."""
    return 50.0 + b_star * math.tan(math.radians(ita))


def test_depth_class_boundaries_are_lower_inclusive() -> None:
    """A value exactly on a boundary belongs to the darker class (lower inclusive)."""
    at_55 = _l_for_ita(55.0)
    assert skin.ita_degrees(at_55, 10.0) == pytest.approx(55.0, abs=1e-6)
    assert skin.depth_class(at_55, 10.0) == "light"
    assert skin.depth_class(at_55 + 0.05, 10.0) == "very-light"

    at_28 = _l_for_ita(28.0)
    assert skin.depth_class(at_28, 10.0) == "tan"
    assert skin.depth_class(at_28 + 0.05, 10.0) == "intermediate"


def test_depth_class_covers_every_enum_value() -> None:
    """Every declared depth class must be reachable."""
    produced = {
        skin.depth_class(_l_for_ita(70.0), 10.0),
        skin.depth_class(_l_for_ita(48.0), 10.0),
        skin.depth_class(_l_for_ita(35.0), 10.0),
        skin.depth_class(_l_for_ita(20.0), 10.0),
        skin.depth_class(_l_for_ita(0.0), 10.0),
        skin.depth_class(_l_for_ita(-45.0), 10.0),
    }
    assert produced == {"very-light", "light", "intermediate", "tan", "brown", "dark"}


def test_undertone_label_matches_the_contract_example() -> None:
    """The contract's example hue gives the contract's example label."""
    result = skin.undertone(62.14, 12.42, 18.91)
    assert result["label"] == "neutral-warm"
    axis = result["axis"]
    assert isinstance(axis, dict)
    assert axis["hueAngleDeg"] == pytest.approx(56.70, abs=0.02)
    assert axis["aOverB"] == pytest.approx(0.657, abs=0.001)


def test_undertone_probabilities_are_a_distribution() -> None:
    """The four class weights must be non-negative and sum to one."""
    result = skin.undertone(62.14, 12.42, 18.91)
    probabilities = result["probabilities"]
    assert isinstance(probabilities, dict)
    assert set(probabilities) == {"cool", "neutral", "warm", "olive"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(value >= 0.0 for value in probabilities.values())


def test_undertone_labels_are_in_the_contract_enum() -> None:
    """Sweeping the hue circle must only ever produce contract labels."""
    allowed = {
        "cool",
        "neutral-cool",
        "neutral",
        "neutral-warm",
        "warm",
        "olive",
    }
    for hue in range(20, 100, 2):
        radians = np.radians(hue)
        a_star = 18.0 * float(np.cos(radians))
        b_star = 18.0 * float(np.sin(radians))
        label = skin.undertone(60.0, a_star, b_star)["label"]
        assert label in allowed


def test_melanin_index_increases_with_depth() -> None:
    """Darker skin must produce a larger melanin index."""
    light, _ = skin.melanin_hemoglobin([0.60, 0.40, 0.30])
    dark, _ = skin.melanin_hemoglobin([0.18, 0.08, 0.045])
    assert dark > light


def test_indices_stay_inside_zero_one() -> None:
    """Both chromophore indices are normalised into 0..1 for any input."""
    for linear in ([0.9, 0.8, 0.7], [0.2, 0.1, 0.05], [0.001, 0.001, 0.001], [1.0, 1.0, 1.0]):
        melanin, hemoglobin = skin.melanin_hemoglobin(linear)
        assert 0.0 <= melanin <= 1.0
        assert 0.0 <= hemoglobin <= 1.0


def test_optical_density_round_trip() -> None:
    """OD -> linear must invert linear -> OD for representable values."""
    linear = np.array([0.5, 0.2, 0.05])
    assert np.allclose(skin.od_to_linear_rgb(skin.linear_rgb_to_od(linear)), linear, atol=1e-12)


def test_optical_density_of_overbright_input_is_negative() -> None:
    """Linear values above 1.0 (physically impossible) give a negative density."""
    assert np.all(skin.linear_rgb_to_od([1.5, 2.0, 1.0]) <= 0.0 + 1e-12)


def test_hue_angle_quadrants() -> None:
    """Hue angles must follow the CIELAB convention (0 = +a, 90 = +b)."""
    assert skin.hue_angle_deg(1.0, 0.0) == pytest.approx(0.0)
    assert skin.hue_angle_deg(0.0, 1.0) == pytest.approx(90.0)
    assert skin.hue_angle_deg(-1.0, 0.0) == pytest.approx(180.0)
    assert skin.hue_angle_deg(0.0, -1.0) == pytest.approx(270.0)


def test_config_ita_classes_are_ordered() -> None:
    """The ITA table must be strictly descending for the lookup to work."""
    bounds = [lower for _, lower in config.ITA_CLASSES]
    assert bounds == sorted(bounds, reverse=True)

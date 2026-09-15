"""CIEDE2000 gate: the official Sharma et al. supplementary test data.

Source: G. Sharma, W. Wu, E. N. Dalal, *The CIEDE2000 Color-Difference Formula:
Implementation Notes, Supplementary Test Data, and Mathematical Observations*,
Color Research and Application 30(1):21-30, 2005 -- the "supplementary test
data" table published with the paper (also distributed as ``testdata.txt``).

The table below is embedded verbatim (four decimals, exactly as published) so the
gate runs offline. Each row is::

    (L1, a1, b1, L2, a2, b2, expected_deltaE00)

All 34 rows are transcribed from the canonical
``ciede2000testdata.txt`` published with the paper.

Reference: G. Sharma, W. Wu, E. N. Dalal, *The CIEDE2000 Color-Difference
Formula: Implementation Notes, Supplementary Test Data, and Mathematical
Observations*, Color Research and Application 30(1):21-30, 2005.
Test data: https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/ciede2000testdata.txt
"""

from __future__ import annotations

import numpy as np
import pytest

from skintone.color.ciede2000 import ciede2000, delta_e00

# (L1, a1, b1, L2, a2, b2, expected dE00)
SHARMA_TEST_DATA: tuple[tuple[float, float, float, float, float, float, float], ...] = (
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
)


def test_dataset_has_at_least_the_official_34_pairs() -> None:
    """The gate is only meaningful with the full published data set present."""
    assert len(SHARMA_TEST_DATA) >= 34


@pytest.mark.parametrize(
    ("lab1", "lab2", "expected"),
    [
        ((row[0], row[1], row[2]), (row[3], row[4], row[5]), row[6])
        for row in SHARMA_TEST_DATA
    ],
    ids=[f"pair-{index:02d}" for index in range(1, len(SHARMA_TEST_DATA) + 1)],
)
def test_sharma_pair(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
    expected: float,
) -> None:
    """Each published pair must match to the precision of the published table."""
    assert delta_e00(lab1, lab2) == pytest.approx(expected, abs=1e-4)


def test_vectorised_matches_scalar() -> None:
    """The array path used for whole cards must agree with the scalar path."""
    first = np.array([row[:3] for row in SHARMA_TEST_DATA], dtype=float)
    second = np.array([row[3:6] for row in SHARMA_TEST_DATA], dtype=float)
    expected = np.array([row[6] for row in SHARMA_TEST_DATA], dtype=float)
    assert np.allclose(ciede2000(first, second), expected, atol=1e-4)


def test_identical_colours_are_zero() -> None:
    """A colour compared with itself has zero difference."""
    assert delta_e00((52.3, 14.2, 18.9), (52.3, 14.2, 18.9)) == pytest.approx(0.0, abs=1e-12)


def test_symmetry() -> None:
    """CIEDE2000 is symmetric in its two arguments."""
    first = (61.2901, 3.7196, -5.3901)
    second = (61.4292, 2.2480, -4.9620)
    assert delta_e00(first, second) == pytest.approx(delta_e00(second, first), abs=1e-12)


def test_parametric_weights_scale_terms() -> None:
    """kL > 1 must shrink a purely lightness-driven difference."""
    base = delta_e00((50.0, 0.0, 0.0), (60.0, 0.0, 0.0))
    damped = float(
        ciede2000(np.array([50.0, 0.0, 0.0]), np.array([60.0, 0.0, 0.0]), k_l=2.0)
    )
    assert damped == pytest.approx(base / 2.0, rel=1e-9)

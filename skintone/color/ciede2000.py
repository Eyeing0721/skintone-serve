"""CIEDE2000 colour difference, implemented from the published formulation.

Reference: G. Sharma, W. Wu, E. N. Dalal, *The CIEDE2000 Color-Difference
Formula: Implementation Notes, Supplementary Test Data, and Mathematical
Observations*, Color Research and Application 30(1), 2005.

The implementation is validated against the official 34-pair supplementary test
data set in ``tests/test_ciede2000.py``; the data is embedded in that test file
so the gate can be re-run without network access.

Units and colour space
----------------------
Inputs are CIELAB triples ``[L*, a*, b*]`` with ``L*`` in 0..100 and ``a*``,
``b*`` roughly -128..127 (D65 white point in this project, but CIEDE2000 itself
is white-point agnostic once both colours share one). The result is a
dimensionless colour difference where ~1.0 is the classic "just noticeable
difference" on a good display and 2.3 corresponds to a typical CMC(1:1) tolerance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: ``25**7`` appears in the chroma-weighting terms; precomputed for speed.
_POW25_7 = 25.0**7


def ciede2000(
    lab1: ArrayLike,
    lab2: ArrayLike,
    k_l: float = 1.0,
    k_c: float = 1.0,
    k_h: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the CIEDE2000 colour difference between two CIELAB colours.

    Args:
        lab1: CIELAB ``[L*, a*, b*]``, trailing axis of length 3 (any shape).
        lab2: CIELAB ``[L*, a*, b*]``, broadcastable against ``lab1``.
        k_l: parametric lightness weight (1.0 for the reference conditions).
        k_c: parametric chroma weight (1.0).
        k_h: parametric hue weight (1.0).

    Returns:
        ``deltaE00`` as a float ndarray broadcast over the leading shape;
        a Python float when both inputs are single colours (0-d result).
    """
    first = np.asarray(lab1, dtype=np.float64)
    second = np.asarray(lab2, dtype=np.float64)
    first, second = np.broadcast_arrays(first, second)

    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + _POW25_7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    delta_lp = l2 - l1
    delta_cp = c2p - c1p

    chroma_product = c1p * c2p
    raw_hue_delta = h2p - h1p
    delta_hp = np.where(
        raw_hue_delta > 180.0,
        raw_hue_delta - 360.0,
        np.where(raw_hue_delta < -180.0, raw_hue_delta + 360.0, raw_hue_delta),
    )
    delta_hp = np.where(chroma_product == 0.0, 0.0, delta_hp)
    delta_big_hp = 2.0 * np.sqrt(chroma_product) * np.sin(np.radians(delta_hp / 2.0))

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    hue_sum = h1p + h2p
    h_bar_p = np.where(
        np.abs(h1p - h2p) <= 180.0,
        hue_sum / 2.0,
        np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
    )
    h_bar_p = np.where(chroma_product == 0.0, hue_sum, h_bar_p)

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    delta_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + _POW25_7))
    s_l = 1.0 + (0.015 * (l_bar_p - 50.0) ** 2) / np.sqrt(20.0 + (l_bar_p - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    term_l = delta_lp / (k_l * s_l)
    term_c = delta_cp / (k_c * s_c)
    term_h = delta_big_hp / (k_h * s_h)

    result = np.sqrt(term_l**2 + term_c**2 + term_h**2 + r_t * term_c * term_h)
    if result.ndim == 0:
        return np.float64(result)
    return np.asarray(result, dtype=np.float64)


def delta_e00(lab1: ArrayLike, lab2: ArrayLike) -> float:
    """Return CIEDE2000 for two single CIELAB colours as a Python float.

    Args:
        lab1: CIELAB ``[L*, a*, b*]``.
        lab2: CIELAB ``[L*, a*, b*]``.

    Returns:
        The colour difference as a float.
    """
    return float(ciede2000(lab1, lab2))


def delta_e00_mean(reference: ArrayLike, measured: ArrayLike) -> float:
    """Mean CIEDE2000 over paired CIELAB rows.

    Args:
        reference: ``(N, 3)`` CIELAB reference values.
        measured: ``(N, 3)`` CIELAB measured values.

    Returns:
        Arithmetic mean of the ``N`` per-pair differences.
    """
    values = ciede2000(reference, measured)
    return float(np.mean(values))


def delta_e00_max(reference: ArrayLike, measured: ArrayLike) -> float:
    """Maximum CIEDE2000 over paired CIELAB rows.

    Args:
        reference: ``(N, 3)`` CIELAB reference values.
        measured: ``(N, 3)`` CIELAB measured values.

    Returns:
        Largest of the ``N`` per-pair differences.
    """
    values = ciede2000(reference, measured)
    return float(np.max(values))

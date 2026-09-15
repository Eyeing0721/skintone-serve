"""Skin descriptors: ITA depth, undertone axes, log-chromaticity decomposition.

Three independent axes are produced (contract section 0):

1. **Depth** -- the dermatological Individual Typology Angle
   ``ITA = atan((L* - 50) / b*)`` in degrees. This is the only axis with a
   published, widely reproduced mapping to clinical measurement, and it is the
   one this project trusts.
2. **Undertone** -- CIELAB hue angle ``h_ab``, the ``a*/b*`` ratio and chroma
   ``C*``. The *label* derived from them is a documented heuristic with a
   measured ceiling on accuracy (see ``docs/ALGORITHM.md``); the raw numbers are
   always returned next to it so a caller can disagree with the label.
3. **Surface state** -- specular and clipping fractions, measured in
   :mod:`skintone.vision.roi`, plus the melanin/haemoglobin indices below.

Colour-space conventions
-----------------------
* Colour inputs are CIELAB (D65): ``L*`` in 0..100, ``a*``/``b*`` roughly
  -128..127, ``h_ab`` in degrees in ``[0, 360)``.
* ``melanin_index`` / ``hemoglobin_index`` are dimensionless and normalised to
  roughly 0..1 by the references in :mod:`skintone.config`. They are *relative*
  chromophore estimates from a single camera response, not concentrations.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import config


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def hue_angle_deg(a: float, b: float) -> float:
    """CIELAB hue angle ``h_ab`` in degrees in ``[0, 360)``.

    Args:
        a: CIELAB ``a*`` (positive = red).
        b: CIELAB ``b*`` (positive = yellow).

    Returns:
        The hue angle; 0 degrees is +a* (red), 90 degrees is +b* (yellow).
    """
    return math.degrees(math.atan2(b, a)) % 360.0


def chroma(a: float, b: float) -> float:
    """CIELAB chroma ``C* = hypot(a*, b*)`` (dimensionless, 0 = neutral)."""
    return math.hypot(a, b)


def ita_degrees(l_star: float, b_star: float) -> float:
    """Individual Typology Angle in degrees.

    ``ITA = atan((L* - 50) / b*) * 180 / pi`` with the four-quadrant arctangent,
    evaluated in CIELAB (D65). Larger values mean lighter skin.

    Args:
        l_star: CIELAB ``L*`` in 0..100.
        b_star: CIELAB ``b*``.

    Returns:
        ITA in degrees, conventionally in -90..90 for real skin.
    """
    return math.degrees(math.atan2(l_star - 50.0, b_star))


def depth_class(l_star: float, b_star: float) -> str:
    """Map ITA to the contract's depth enum (lower bound inclusive).

    Args:
        l_star: CIELAB ``L*`` in 0..100.
        b_star: CIELAB ``b*``.

    Returns:
        One of ``very-light``, ``light``, ``intermediate``, ``tan``, ``brown``,
        ``dark`` (contract section 2.3).
    """
    ita = ita_degrees(l_star, b_star)
    for name, lower in config.ITA_CLASSES:
        if ita > lower:
            return name
    return config.ITA_CLASSES[-1][0]


def warm_cool_axis(hue_angle: float) -> float:
    """Project ``h_ab`` onto the warm/cool axis in ``[-1, 1]``.

    Golden/yellow skin sits at a larger ``h_ab`` than pink/rosy skin, so the axis
    grows with the hue angle. ``+1`` = warm, ``-1`` = cool, ``0`` = neutral.

    Args:
        hue_angle: CIELAB hue angle in degrees.

    Returns:
        The axis value in ``[-1, 1]``.
    """
    return math.tanh(
        (hue_angle - config.UNDERTONE_HUE_NEUTRAL_DEG) / config.UNDERTONE_HUE_AXIS_SCALE_DEG
    )


def undertone_axis_label(axis: float) -> str:
    """Turn the warm/cool axis into a contract undertone label.

    Args:
        axis: value from :func:`warm_cool_axis`, in ``[-1, 1]``.

    Returns:
        One of ``warm``, ``neutral-warm``, ``neutral``, ``neutral-cool``, ``cool``.
    """
    for name, cut in config.UNDERTONE_LABEL_CUTS:
        if axis >= cut:
            return name
    return config.UNDERTONE_COOL_LABEL


def olive_evidence(hue_angle: float, a_over_b: float | None, chroma_value: float) -> float:
    """Heuristic olive undertone evidence in ``[0, 1]``.

    Olive skin is modelled as *high* hue angle (yellow-green), a *low* ``a*/b*``
    ratio (relatively less red) and *low* chroma. Each factor is a logistic ramp;
    their product is the evidence.

    Args:
        hue_angle: CIELAB hue angle in degrees.
        a_over_b: ``a*/b*`` ratio, or ``None`` when ``b*`` is near zero.
        chroma_value: CIELAB ``C*``.

    Returns:
        Evidence in ``[0, 1]``; it is an input to :func:`undertone`, not a
        calibrated probability.
    """
    if a_over_b is None:
        ratio_term = 0.0
    else:
        ratio_term = _sigmoid(
            (config.UNDERTONE_OLIVE_RATIO_CENTER - a_over_b)
            / config.UNDERTONE_OLIVE_RATIO_SCALE
        )
    hue_term = _sigmoid(
        (hue_angle - config.UNDERTONE_OLIVE_HUE_CENTER) / config.UNDERTONE_OLIVE_HUE_SCALE
    )
    chroma_term = _sigmoid(
        (config.UNDERTONE_OLIVE_CHROMA_CENTER - chroma_value)
        / config.UNDERTONE_OLIVE_CHROMA_SCALE
    )
    return float(hue_term * ratio_term * chroma_term)


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a small labelled dict."""
    peak = max(logits.values())
    exponents = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(exponents.values())
    return {key: value / total for key, value in exponents.items()}


def undertone(l_star: float, a_star: float, b_star: float) -> dict[str, object]:
    """Classify undertone and expose the numbers behind the label.

    Args:
        l_star: CIELAB ``L*`` (used only for context; the axis is hue driven).
        a_star: CIELAB ``a*``.
        b_star: CIELAB ``b*``.

    Returns:
        A dict with:

        * ``label`` -- one of the contract's six undertone labels.
        * ``axis`` -- ``{"aOverB": float|None, "hueAngleDeg": float}``, the raw
          axes so a caller can recompute or disagree with the label.
        * ``probabilities`` -- ``{"cool", "neutral", "warm", "olive"}`` summing to
          1. These are **heuristic class weights** derived from a softmax over
          distance to declared prototypes, not a calibrated posterior.
    """
    hue = hue_angle_deg(a_star, b_star)
    chroma_value = chroma(a_star, b_star)
    a_over_b = None if abs(b_star) < 1e-6 else a_star / b_star
    axis = warm_cool_axis(hue)

    evidence = olive_evidence(hue, a_over_b, chroma_value)
    logits = {
        name: -((axis - prototype) ** 2) / config.UNDERTONE_SOFTMAX_TAU
        for name, prototype in config.UNDERTONE_AXIS_PROTOTYPES.items()
    }
    logits["olive"] = (
        config.UNDERTONE_OLIVE_LOGIT_GAIN * evidence + config.UNDERTONE_OLIVE_LOGIT_BIAS
    )
    probabilities = _softmax(logits)

    label = undertone_axis_label(axis)
    if (
        evidence >= config.UNDERTONE_OLIVE_LABEL_MIN_EVIDENCE
        and probabilities["olive"] >= config.UNDERTONE_OLIVE_LABEL_MIN_PROB
    ):
        label = "olive"

    return {
        "label": label,
        "axis": {"aOverB": a_over_b, "hueAngleDeg": hue},
        "probabilities": probabilities,
    }


def linear_rgb_to_od(linear_rgb: ArrayLike) -> NDArray[np.float64]:
    """Convert **linear** sRGB (0..1) to optical density.

    ``OD = -log(linear)`` per the contract's section 5C, with ``log`` the natural
    logarithm and a floor applied so that black pixels stay finite. For a single
    channel, ``c = exp(-OD)``, i.e. Beer-Lambert absorption.

    Args:
        linear_rgb: linear sRGB in 0..1, trailing axis of length 3 (never encoded
            sRGB -- taking a logarithm of gamma-encoded values is meaningless).

    Returns:
        Optical density, same shape, dimensionless and >= 0 for input <= 1.
    """
    value = np.clip(np.asarray(linear_rgb, dtype=np.float64), config.OD_LINEAR_FLOOR, None)
    return -np.log(value)


def od_to_linear_rgb(od: ArrayLike) -> NDArray[np.float64]:
    """Invert :func:`linear_rgb_to_od`: linear sRGB (0..1) from optical density."""
    return np.exp(-np.asarray(od, dtype=np.float64))


def _melanin_axis() -> NDArray[np.float64]:
    """Melanin direction in linear-sRGB OD space (unit length).

    Melanin absorbs broadly with a monotone, power-law extinction, so the least
    contaminated read-out is the red band, where oxygenated haemoglobin is nearly
    transparent.
    """
    axis = np.asarray(config.OD_MELANIN_AXIS, dtype=np.float64)
    return axis / np.linalg.norm(axis)


def _hemoglobin_axis() -> NDArray[np.float64]:
    """Haemoglobin direction in linear-sRGB OD space (unit length).

    Oxygenated haemoglobin has its visible Q bands near 542 nm and 577 nm, so the
    signal is the green band measured against the red band as a scattering
    baseline. The returned direction is orthonormal to :func:`_melanin_axis`.
    """
    axis = np.asarray(config.OD_HEMOGLOBIN_AXIS, dtype=np.float64)
    return axis / np.linalg.norm(axis)


def melanin_hemoglobin(linear_rgb: ArrayLike) -> tuple[float, float]:
    """Decompose skin colour into melanin and haemoglobin indices.

    The optical density of skin is projected onto two orthonormal directions in
    linear-sRGB OD space::

        melanin     ~= <OD, e_melanin>       (red-band density)
        haemoglobin ~= <OD, e_haemoglobin>   (green-band excess over red)

    Both projections are clipped at zero (a chromophore cannot have negative
    density) and normalised by the references in :mod:`skintone.config`.

    Args:
        linear_rgb: **linear** sRGB in 0..1 for one pixel or the mean of an ROI.

    Returns:
        ``(melanin_index, hemoglobin_index)``, each clipped to ``[0, 1]`` and
        dimensionless. These are *relative, directional* estimates from a single
        camera response, and must not be read as concentrations. Their value is
        that they are far less sensitive to shadow and specular variation than
        raw RGB. With only three broad camera bands the two chromophores are not
        cleanly separable; see ``docs/ALGORITHM.md`` for the measured limit.
    """
    od = linear_rgb_to_od(linear_rgb).reshape(3)
    melanin_raw = float(np.dot(od, _melanin_axis()))
    hemoglobin_raw = float(np.dot(od, _hemoglobin_axis()))
    melanin = max(melanin_raw, 0.0) / config.OD_MELANIN_REFERENCE
    hemoglobin = max(hemoglobin_raw, 0.0) / config.OD_HEMOGLOBIN_REFERENCE
    return float(np.clip(melanin, 0.0, 1.0)), float(np.clip(hemoglobin, 0.0, 1.0))

"""Garment-colour advice, generated from measurable CIELAB rules (contract 6).

Every recommendation is the result of a stated numeric rule applied to the
measured skin colour -- no seasonal-colour labels, no taste claims:

* **recommended**: CIELAB hue offset from the skin hue ``h_ab`` held inside
  25-60 degrees (harmonious, not matching), chroma ratio to skin ``C*`` inside
  0.8-1.6, lightness placed at a target contrast against skin ``L*``.
* **avoid**: hue within 15 degrees of skin with a much higher chroma (reads as
  muddy), or nearly the same lightness as skin with a very low chroma (reads as
  grey).

Colour space
------------
All inputs are CIELAB (D65, ``L*`` 0..100). All outputs are 8-bit sRGB hex plus
the CIELAB actually used, after gamut clipping to the sRGB cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import config
from .srgb import lab_to_lch, lab_to_linear_rgb, linear_to_hex, linear_to_srgb8, lch_to_lab

#: Coarse hue-angle names in Chinese, keyed by the inclusive lower bound in degrees.
_HUE_NAMES: tuple[tuple[float, str], ...] = (
    (0.0, "红"),
    (15.0, "橙"),
    (45.0, "驼"),
    (70.0, "黄绿"),
    (105.0, "绿"),
    (160.0, "青"),
    (200.0, "蓝"),
    (260.0, "紫"),
    (300.0, "玫红"),
    (345.0, "红"),
)


def _hue_name(hue_deg: float) -> str:
    """Return a coarse Chinese hue name for a CIELAB hue angle in degrees."""
    name = _HUE_NAMES[0][1]
    for lower, candidate in _HUE_NAMES:
        if hue_deg >= lower:
            name = candidate
        else:
            break
    return name


def _colour_name(lab: np.ndarray) -> str:
    """Describe a CIELAB colour with a lightness/chroma qualifier plus hue name."""
    lightness, chroma_value, hue = (float(v) for v in lab_to_lch(lab))
    hue_name = _hue_name(hue)
    if chroma_value < 12.0:
        return f"灰{hue_name}"
    if lightness < 35.0:
        return f"深{hue_name}"
    if lightness > 78.0:
        return f"浅{hue_name}"
    return hue_name


def clip_to_gamut(lab: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Reduce chroma until a CIELAB colour fits inside the sRGB gamut.

    Lightness and hue are preserved; only ``C*`` is scaled down, which keeps the
    hue relationship that the recommendation rules depend on.

    Args:
        lab: CIELAB ``[L*, a*, b*]`` (D65).
        tolerance: relative chroma resolution of the bisection.

    Returns:
        CIELAB ``[L*, a*, b*]`` inside the sRGB gamut (component values in 0..1
        after conversion).
    """
    lab = np.asarray(lab, dtype=np.float64)
    linear = lab_to_linear_rgb(lab)
    if np.all(linear >= -1e-9) and np.all(linear <= 1.0 + 1e-9):
        return lab
    lightness, chroma_value, hue = (float(v) for v in lab_to_lch(lab))
    low, high = 0.0, chroma_value
    best = np.array([lightness, 0.0, 0.0], dtype=np.float64)
    for _ in range(40):
        mid = 0.5 * (low + high)
        candidate = lch_to_lab(np.array([lightness, mid, hue], dtype=np.float64))
        linear = lab_to_linear_rgb(candidate)
        if np.all(linear >= -1e-9) and np.all(linear <= 1.0 + 1e-9):
            best = candidate
            low = mid
        else:
            high = mid
        if high - low <= tolerance * max(chroma_value, 1.0):
            break
    return best


def _lab_to_hex(lab: np.ndarray) -> str:
    """Convert an in-gamut CIELAB colour to a lowercase ``#rrggbb`` string."""
    return linear_to_hex(np.clip(lab_to_linear_rgb(lab), 0.0, 1.0))


def contrast_level(hair_l_star: float | None, skin_l_star: float) -> tuple[str, str | None]:
    """Classify the skin/hair contrast from the lightness difference.

    Args:
        hair_l_star: CIELAB ``L*`` of the hair, or ``None`` when the client did
            not supply one.
        skin_l_star: CIELAB ``L*`` of the measured skin.

    Returns:
        ``(level, warning)``. Without a hair colour the level falls back to a
        single-axis estimate from skin lightness alone and ``warning`` explains
        that, per contract section 6.
    """
    if hair_l_star is None:
        if skin_l_star >= 70.0 or skin_l_star <= 35.0:
            level = "high"
        elif skin_l_star >= 52.0 or skin_l_star <= 45.0:
            level = "medium"
        else:
            level = "low"
        return level, "未提供头发色，对比度建议基于肤色单轴估计"
    difference = abs(hair_l_star - skin_l_star)
    for name, cut in config.CONTRAST_LEVEL_CUTS:
        if difference >= cut:
            return name, None
    return config.CONTRAST_LEVEL_LOW, None


def _palette_specs(chroma_value: float) -> list[tuple[float, float, float]]:
    """Return ``(hue_offset_deg, chroma_ratio, lightness_delta)`` candidates.

    Every hue offset stays inside the contract's 25-60 degree harmony window, in
    both directions, and every chroma ratio stays inside 0.8-1.6. The widest hue
    separations are paired with the largest lightness moves so the palette spans
    both a subtle and a bold option.
    """
    chroma_value = max(chroma_value, 1.0)
    windows = (
        (config.ADVICE_HUE_OFFSET_MIN_DEG, config.ADVICE_CHROMA_RATIO_MIN),
        (-config.ADVICE_HUE_OFFSET_MIN_DEG, config.ADVICE_CHROMA_RATIO_MIN + 0.2),
        (
            (config.ADVICE_HUE_OFFSET_MIN_DEG + config.ADVICE_HUE_OFFSET_MAX_DEG) / 2.0,
            config.ADVICE_CHROMA_RATIO_MIN + 0.4,
        ),
        (
            -(config.ADVICE_HUE_OFFSET_MIN_DEG + config.ADVICE_HUE_OFFSET_MAX_DEG) / 2.0,
            config.ADVICE_CHROMA_RATIO_MIN + 0.6,
        ),
        (config.ADVICE_HUE_OFFSET_MAX_DEG, config.ADVICE_CHROMA_RATIO_MAX),
        (-config.ADVICE_HUE_OFFSET_MAX_DEG, config.ADVICE_CHROMA_RATIO_MAX),
    )
    deltas = config.ADVICE_LIGHTNESS_DELTA_TARGETS
    specs: list[tuple[float, float, float]] = []
    for index, (offset, ratio) in enumerate(windows):
        specs.append((offset, min(ratio, config.ADVICE_CHROMA_RATIO_MAX), deltas[index % len(deltas)]))
    return specs


def build_palette(lab: np.ndarray, limit: int = 6) -> list[dict[str, Any]]:
    """Generate recommended garment colours from the measured skin colour.

    Args:
        lab: CIELAB ``[L*, a*, b*]`` of the skin (D65).
        limit: Maximum number of recommendations to return.

    Returns:
        A list of ``{"hex": str, "lab": [L*, a*, b*], "name": str}`` entries, each
        derived by one :func:`_palette_specs` rule and gamut-clipped with hue and
        lightness preserved.
    """
    lightness, chroma_value, hue = (float(v) for v in lab_to_lch(lab))
    results: list[dict[str, Any]] = []
    for offset, ratio, delta in _palette_specs(chroma_value):
        target = lch_to_lab(
            np.array(
                [
                    float(np.clip(lightness + delta, 12.0, 94.0)),
                    min(chroma_value * ratio, 90.0),
                    (hue + offset) % 360.0,
                ],
                dtype=np.float64,
            )
        )
        target = clip_to_gamut(target)
        results.append(
            {
                "hex": _lab_to_hex(target),
                "lab": [round(float(v), 2) for v in target],
                "name": _colour_name(target),
            }
        )
        if len(results) >= limit:
            break
    return results


def build_avoid(lab: np.ndarray, limit: int = 3) -> list[dict[str, str]]:
    """Generate colours to avoid, each with the numeric rule that rejected it.

    Args:
        lab: CIELAB ``[L*, a*, b*]`` of the skin (D65).
        limit: Maximum number of entries to return.

    Returns:
        A list of ``{"hex": str, "reason": str}`` entries.
    """
    lightness, chroma_value, hue = (float(v) for v in lab_to_lch(lab))
    candidates: list[tuple[np.ndarray, str]] = []

    muddy = clip_to_gamut(
        lch_to_lab(
            np.array(
                [
                    float(np.clip(lightness, 12.0, 94.0)),
                    min(chroma_value * config.ADVICE_AVOID_CHROMA_RATIO_MIN * 1.4, 95.0),
                    (hue + 6.0) % 360.0,
                ],
                dtype=np.float64,
            )
        )
    )
    candidates.append(
        (
            muddy,
            f"色相与肤色仅差 6°（阈值 {config.ADVICE_AVOID_HUE_DELTA_MAX_DEG:.0f}°）"
            f"且彩度约为肤色的 {config.ADVICE_AVOID_CHROMA_RATIO_MIN * 1.4:.1f} 倍，显脏",
        )
    )

    grey = lch_to_lab(
        np.array([float(np.clip(lightness, 12.0, 94.0)), config.ADVICE_AVOID_CHROMA_MAX - 2.0, (hue + 20.0) % 360.0])
    )
    candidates.append(
        (
            grey,
            f"明度与肤色相差不足 {config.ADVICE_AVOID_LIGHTNESS_DELTA_MAX:.0f} 且彩度低于"
            f" {config.ADVICE_AVOID_CHROMA_MAX:.0f}，显灰",
        )
    )

    dark_grey = lch_to_lab(np.array([max(lightness - 4.0, 8.0), 6.0, (hue + 200.0) % 360.0]))
    candidates.append(
        (
            dark_grey,
            "低彩度深色贴近肤色明度，缺乏对比，显黯淡",
        )
    )

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for colour, reason in candidates:
        hex_value = _lab_to_hex(clip_to_gamut(colour))
        if hex_value in seen:
            continue
        seen.add(hex_value)
        results.append({"hex": hex_value, "reason": reason})
        if len(results) >= limit:
            break
    return results


def build_advice(lab: np.ndarray, hair_l_star: float | None) -> tuple[dict[str, Any], list[str]]:
    """Assemble the ``advice`` block and any warnings it produced.

    Args:
        lab: CIELAB ``[L*, a*, b*]`` of the skin (D65).
        hair_l_star: CIELAB ``L*`` of the hair, or ``None``.

    Returns:
        ``(advice, warnings)`` where ``advice`` matches the contract's ``advice``
        object and ``warnings`` are human-readable caveats to append.
    """
    lightness = float(lab[0])
    level, warning = contrast_level(hair_l_star, lightness)
    advice = {
        "contrastLevel": level,
        "palette": build_palette(lab),
        "avoid": build_avoid(lab),
    }
    return advice, ([warning] if warning else [])


def srgb8_of_hex(value: str) -> list[int]:
    """Return the 8-bit sRGB triple of a ``#rrggbb`` string (helper for tests)."""
    from .srgb import hex_to_srgb8

    return [int(round(float(v))) for v in hex_to_srgb8(value)]


def round_trip_error(lab: np.ndarray) -> float:
    """Return the 8-bit sRGB round-trip error of a CIELAB colour.

    Useful for asserting that a generated palette entry survives quantisation.

    Args:
        lab: CIELAB ``[L*, a*, b*]``.

    Returns:
        Largest absolute per-channel 8-bit error after a Lab -> sRGB -> Lab trip.
    """
    from .srgb import srgb8_to_lab

    clipped = clip_to_gamut(lab)
    srgb8 = linear_to_srgb8(np.clip(lab_to_linear_rgb(clipped), 0.0, 1.0))
    back = srgb8_to_lab(srgb8)
    return float(np.max(np.abs(back - clipped)))

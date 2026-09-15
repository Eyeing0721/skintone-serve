"""Illuminant estimation support: CCT, Planckian and CIE daylight loci.

Everything here is deliberately **one-dimensional**. Estimating an illuminant as
a free point in chromaticity space is unstable from a single photograph; instead
this module estimates a chromaticity, converts it to a correlated colour
temperature, and *projects* it onto a physically possible locus, leaving a single
unknown (the CCT) plus a small residual (Duv).

Methods
-------
* CCT from chromaticity: McCamy's cubic approximation, used only to seed the
  search (:func:`cct_mccamy`).
* Planckian locus: Planck's law integrated against analytic CIE 1931
  2-degree colour matching functions. The CMFs use the multi-lobe piecewise
  Gaussian fit of Wyman, Sloan & Shirley (JCGT 2013), which keeps this module
  free of large tables while staying well inside the tolerance needed here.
* CIE daylight locus: the official CIE D-series closed form (:func:`daylight_xy`).
* Projection: nearest point on a locus in CIE 1960 UCS ``uv``, giving a refined
  CCT and a signed perpendicular offset ``Duv``.

Units
-----
* wavelength ``lambda``: nanometres, sampled 360..830 nm.
* temperature: kelvin.
* ``xy``: CIE 1931 chromaticity.
* ``uv``: CIE 1960 UCS, ``u = 4X/(X+15Y+3Z)``, ``v = 6Y/(X+15Y+3Z)``.
* ``duv``: signed distance in ``uv`` units; positive = above the locus (greenish),
  negative = below (pinkish/magenta). ``|Duv| > 0.02`` is treated as unreliable
  (contract section 5B).
* ``mired``: micro reciprocal degrees, ``1e6 / T``; used because it is the
  perceptually uniform axis along the locus.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import config
from .srgb import D65_CCT_K, D65_XY, xyz_to_xy

# ---------------------------------------------------------------------------
# Analytic CIE 1931 2-degree colour matching functions
# ---------------------------------------------------------------------------

#: Multi-lobe piecewise-Gaussian fit parameters: (mu, sigma_short, sigma_long).
_CMF_X = ((599.8, 37.9, 31.0), (442.0, 16.0, 26.7), (501.1, 20.4, 26.2))
_CMF_X_AMP = (1.056, 0.362, -0.065)
_CMF_Y = ((568.8, 46.9, 40.5), (530.9, 16.3, 31.1))
_CMF_Y_AMP = (0.821, 0.286)
_CMF_Z = ((437.0, 11.8, 36.0), (459.0, 26.0, 13.8))
_CMF_Z_AMP = (1.217, 0.681)

WAVELENGTH_MIN_NM = 360.0
WAVELENGTH_MAX_NM = 830.0
WAVELENGTH_STEP_NM = 1.0

#: Planck constants (SI) and speed of light, for spectral radiance in wavelength.
_PLANCK_H = 6.62607015e-34
_PLANCK_C = 2.99792458e8
_BOLTZMANN_K = 1.380649e-23


def _piecewise_gaussian(
    wavelength: NDArray[np.float64],
    mu: float,
    sigma_short: float,
    sigma_long: float,
) -> NDArray[np.float64]:
    """Evaluate one asymmetric Gaussian lobe of the CMF fit.

    Args:
        wavelength: wavelengths in nm.
        mu: peak wavelength in nm.
        sigma_short: width below the peak, in nm.
        sigma_long: width above the peak, in nm.

    Returns:
        Dimensionless lobe values, same shape as ``wavelength``.
    """
    sigma = np.where(wavelength < mu, sigma_short, sigma_long)
    return np.exp(-0.5 * ((wavelength - mu) / sigma) ** 2)


def _cmf_tables(
    wavelength: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return analytic ``x_bar``, ``y_bar``, ``z_bar`` at the given wavelengths.

    Args:
        wavelength: wavelengths in nm.

    Returns:
        Triple of CMF arrays (dimensionless, normalised so that the integral of
        ``y_bar`` over the visible range equals the usual 106.857 scale factor).
    """
    x_bar = sum(
        amp * _piecewise_gaussian(wavelength, *lobe)
        for amp, lobe in zip(_CMF_X_AMP, _CMF_X)
    )
    y_bar = sum(
        amp * _piecewise_gaussian(wavelength, *lobe)
        for amp, lobe in zip(_CMF_Y_AMP, _CMF_Y)
    )
    z_bar = sum(
        amp * _piecewise_gaussian(wavelength, *lobe)
        for amp, lobe in zip(_CMF_Z_AMP, _CMF_Z)
    )
    return x_bar, y_bar, z_bar


@lru_cache(maxsize=1)
def _wavelength_grid() -> NDArray[np.float64]:
    """Cached 1 nm wavelength grid in nm."""
    return np.arange(
        WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM + WAVELENGTH_STEP_NM, WAVELENGTH_STEP_NM
    )


def planck_spectral_radiance(
    wavelength_nm: ArrayLike, temperature_k: float
) -> NDArray[np.float64]:
    """Planck spectral radiance of a black body, per unit wavelength.

    Args:
        wavelength_nm: wavelengths in nanometres (in vacuo).
        temperature_k: black-body temperature in kelvin.

    Returns:
        Spectral radiance in W/(m^2 sr m). Only ratios matter downstream; the
        absolute scale is irrelevant to chromaticity.
    """
    wavelength_m = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    numerator = 2.0 * _PLANCK_H * _PLANCK_C**2
    exponent = _PLANCK_H * _PLANCK_C / (wavelength_m * _BOLTZMANN_K * temperature_k)
    # expm1 keeps precision for the long-wavelength tail; clip the exponent to
    # avoid overflow for very low temperatures.
    exponent = np.clip(exponent, 0.0, 700.0)
    return numerator / (wavelength_m**5 * np.expm1(exponent))


def planck_xyz(temperature_k: float) -> NDArray[np.float64]:
    """CIE XYZ of a Planckian radiator at ``temperature_k``.

    Args:
        temperature_k: black-body temperature in kelvin.

    Returns:
        CIE XYZ normalised so that ``Y == 1``.
    """
    wavelength = _wavelength_grid()
    x_bar, y_bar, z_bar = _cmf_tables(wavelength)
    radiance = planck_spectral_radiance(wavelength, temperature_k)
    x = float(np.sum(radiance * x_bar))
    y = float(np.sum(radiance * y_bar))
    z = float(np.sum(radiance * z_bar))
    if y <= 0:
        raise ValueError(f"degenerate Planckian spectrum at {temperature_k} K")
    return np.array([x / y, 1.0, z / y], dtype=np.float64)


def planck_xy(temperature_k: float) -> NDArray[np.float64]:
    """CIE 1931 chromaticity of the Planckian locus at ``temperature_k``."""
    return xyz_to_xy(planck_xyz(temperature_k))


def daylight_xy(temperature_k: float) -> NDArray[np.float64]:
    """CIE 1931 chromaticity of the CIE D-series daylight locus.

    Uses the official CIE closed form (valid 4000..25000 K; values below 4000 K
    are clamped to the 4000 K endpoint because the daylight locus is undefined
    there).

    Args:
        temperature_k: correlated colour temperature in kelvin.

    Returns:
        ``(x, y)`` chromaticity of the daylight illuminant ``D_T``.
    """
    t = float(np.clip(temperature_k, 4000.0, 25000.0))
    if t <= 7000.0:
        x = (
            -4.6070e9 / t**3
            + 2.9678e6 / t**2
            + 0.09911e3 / t
            + 0.244063
        )
    else:
        x = (
            -2.0064e9 / t**3
            + 1.9018e6 / t**2
            + 0.24748e3 / t
            + 0.237040
        )
    y = -3.000 * x**2 + 2.870 * x - 0.275
    return np.array([x, y], dtype=np.float64)


def planck_uv(temperature_k: float) -> NDArray[np.float64]:
    """Planckian locus point in CIE 1960 UCS ``uv`` at ``temperature_k``."""
    return xy_to_uv(planck_xy(temperature_k))


def daylight_uv(temperature_k: float) -> NDArray[np.float64]:
    """Daylight locus point in CIE 1960 UCS ``uv`` at ``temperature_k``."""
    return xy_to_uv(daylight_xy(temperature_k))


# ---------------------------------------------------------------------------
# xy <-> uv (CIE 1960 UCS)
# ---------------------------------------------------------------------------


def xy_to_uv(xy: ArrayLike) -> NDArray[np.float64]:
    """Convert CIE 1931 ``xy`` to CIE 1960 UCS ``uv``.

    Args:
        xy: chromaticity ``(x, y)``, trailing axis of length 2.

    Returns:
        ``(u, v)`` with ``u = 4x / (-2x + 12y + 3)`` and
        ``v = 6y / (-2x + 12y + 3)``. The 1960 ``v`` is *not* the 1976 ``v'``.
    """
    value = np.asarray(xy, dtype=np.float64)
    x, y = value[..., 0], value[..., 1]
    denominator = -2.0 * x + 12.0 * y + 3.0
    safe = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return np.stack([4.0 * x / safe, 6.0 * y / safe], axis=-1)


def uv_to_xy(uv: ArrayLike) -> NDArray[np.float64]:
    """Convert CIE 1960 UCS ``uv`` back to CIE 1931 ``xy``."""
    value = np.asarray(uv, dtype=np.float64)
    u, v = value[..., 0], value[..., 1]
    denominator = 2.0 * u - 8.0 * v + 4.0
    safe = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return np.stack([3.0 * u / safe, 2.0 * v / safe], axis=-1)


# ---------------------------------------------------------------------------
# CCT
# ---------------------------------------------------------------------------


def cct_mccamy(x: float, y: float) -> float:
    """McCamy's cubic approximation of the correlated colour temperature.

    Args:
        x: CIE 1931 chromaticity x.
        y: CIE 1931 chromaticity y.

    Returns:
        CCT in kelvin. This is only an approximation (a good seed for the locus
        projection, not a final answer); :func:`project_to_locus` refines it.
    """
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33


def _locus_grid(locus: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cached ``(temperatures_k, uv)`` samples of a locus.

    Args:
        locus: ``"planckian"`` or ``"daylight"``.

    Returns:
        Pair of arrays: temperatures in kelvin (ascending) and their ``uv``.
    """
    return _locus_grid_cached(locus)


@lru_cache(maxsize=4)
def _locus_grid_cached(locus: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cached implementation of :func:`_locus_grid`."""
    if locus == "planckian":
        temps = np.geomspace(
            config.CCT_SEARCH_MIN_K, config.CCT_SEARCH_MAX_K, config.CCT_SEARCH_COARSE_STEPS
        )
        uv = np.array([planck_uv(float(t)) for t in temps])
    elif locus == "daylight":
        temps = np.geomspace(4000.0, 25000.0, config.CCT_SEARCH_COARSE_STEPS)
        uv = np.array([daylight_uv(float(t)) for t in temps])
    else:
        raise ValueError(f"unknown locus {locus!r}")
    return temps, uv


def project_to_locus(
    xy: ArrayLike, locus: str = "planckian"
) -> tuple[float, float]:
    """Project a chromaticity onto a locus, returning ``(cct, duv)``.

    The projection is the nearest point on the locus in CIE 1960 UCS ``uv``,
    refined by a local golden-section-free ternary search over temperature.

    Args:
        xy: chromaticity ``(x, y)`` of the illuminant estimate.
        locus: ``"planckian"`` or ``"daylight"``.

    Returns:
        ``(cct_kelvin, duv)`` where ``duv`` is signed: positive when the input
        sits *above* the locus in ``uv`` (greenish), negative when below
        (pinkish). Ordering along the locus is monotonic in temperature, so the
        sign is unambiguous.

    Raises:
        ValueError: if ``locus`` is unknown.
    """
    target = xy_to_uv(np.asarray(xy, dtype=np.float64).reshape(2))
    temps, uv = _locus_grid(locus)

    distances = np.linalg.norm(uv - target, axis=1)
    best = int(np.argmin(distances))

    low = temps[max(best - 1, 0)]
    high = temps[min(best + 1, len(temps) - 1)]
    for _ in range(config.CCT_SEARCH_REFINE_STEPS):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        d_left = np.linalg.norm(_locus_uv(locus, float(left)) - target)
        d_right = np.linalg.norm(_locus_uv(locus, float(right)) - target)
        if d_left < d_right:
            high = right
        else:
            low = left
    cct = 0.5 * (low + high)
    locus_point = _locus_uv(locus, float(cct))
    offset = target - locus_point
    distance = float(np.linalg.norm(offset))
    # Positive Duv above the locus: compare v at (approximately) equal u.
    sign = 1.0 if offset[1] >= 0.0 else -1.0
    return float(cct), float(sign * distance)


def _locus_uv(locus: str, temperature_k: float) -> NDArray[np.float64]:
    """Evaluate one locus at one temperature, in CIE 1960 ``uv``."""
    return planck_uv(temperature_k) if locus == "planckian" else daylight_uv(temperature_k)


def cct_and_duv(
    xy: ArrayLike, locus: str | None = None
) -> tuple[float, float, str]:
    """Return ``(cct, duv, locus_used)`` for a chromaticity.

    Args:
        xy: chromaticity ``(x, y)``.
        locus: force ``"planckian"`` or ``"daylight"``; ``None`` evaluates both
            and keeps the better-fitting one.

    Returns:
        ``(cct_kelvin, duv, locus_name)``.

    Raises:
        ValueError: if ``locus`` is neither ``None`` nor a known locus name.
    """
    candidates = [locus] if locus else ["planckian", "daylight"]
    best: tuple[float, float, str] | None = None
    for name in candidates:
        cct, duv = project_to_locus(xy, name)
        if best is None or abs(duv) < abs(best[1]):
            best = (cct, duv, name)
    assert best is not None
    return best


def mired(temperature_k: float) -> float:
    """Convert kelvin to mired (micro reciprocal degrees)."""
    return 1e6 / float(temperature_k)


def from_mired(value: float) -> float:
    """Convert mired back to kelvin."""
    return 1e6 / float(value)


def locus_xy_at_cct(temperature_k: float, locus: str) -> NDArray[np.float64]:
    """Return the ``xy`` chromaticity of a locus at a given CCT."""
    return _locus_uv_xy(locus, temperature_k)


def _locus_uv_xy(locus: str, temperature_k: float) -> NDArray[np.float64]:
    """Chromaticity ``xy`` of ``locus`` at ``temperature_k``."""
    if locus == "planckian":
        return planck_xy(temperature_k)
    return daylight_xy(temperature_k)


def white_xyz_from_xy(xy: ArrayLike, luminance: float = 1.0) -> NDArray[np.float64]:
    """Return an illuminant white point as CIE XYZ.

    Args:
        xy: CIE 1931 chromaticity ``(x, y)``.
        luminance: Value of ``Y``; only the ratio to D65's ``Y`` matters for
            chromatic adaptation, so the default is fine.

    Returns:
        XYZ triple with ``Y == luminance``.
    """
    x, y = (float(value) for value in np.asarray(xy, dtype=np.float64).reshape(2))
    if abs(y) < 1e-12:
        raise ValueError("degenerate chromaticity with y == 0")
    return np.array([x / y * luminance, luminance, (1.0 - x - y) / y * luminance], dtype=np.float64)


def solve_illuminant(
    xy_estimate: ArrayLike | None,
    method: str,
    illuminant_guess: str = "unknown",
    prior_weight: float | None = None,
    assumed_d65: bool = False,
) -> dict[str, object]:
    """Turn a measured chromaticity into a one-dimensional illuminant estimate.

    The estimate is *projected onto a physically possible locus* (Planckian or
    CIE daylight), which collapses a two-parameter fit into one, and then blended
    with the user's ``capture.illuminantGuess`` prior in **mired** (1e6 / K),
    because mired is the perceptually uniform axis along the locus.

    Args:
        xy_estimate: Measured CIE 1931 chromaticity, or ``None`` to fall back to
            D65 entirely (``assumed_d65`` should then be True).
        method: Label for the response's ``illuminant.method`` field.
        illuminant_guess: One of the contract's ``illuminantGuess`` values.
        prior_weight: 0..1 weight of the prior. ``None`` selects
            ``ILLUMINANT_PRIOR_WEIGHT`` for a known guess and 0 for ``unknown``.
        assumed_d65: Whether the returned illuminant is simply D65.

    Returns:
        A dict with ``method``, ``cct`` (kelvin), ``duv`` (signed CIE 1960 uv
        offset), ``xy`` (the *corrected* chromaticity, i.e. the locus point at the
        final CCT), ``assumedD65``, ``locus``, plus ``priorApplied`` /
        ``priorWeight`` so callers can tell whether the user's guess actually
        changed the result.

    Raises:
        ValueError: if ``xy_estimate`` is ``None`` and ``assumed_d65`` is False.
    """
    if xy_estimate is None:
        if not assumed_d65:
            raise ValueError("xy_estimate is required unless assumed_d65 is True")
        # 一个中性参考面都采不到时（无眼白 / 牙齿 / 背景候选），用户选的光源就是
        # 唯一可用的信息。此时**不能一刀切假定 D65**：白炽灯下拍的照片按日光处理
        # 会明显偏色。用更高的权重（ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK）把先验
        # 和 D65 混合——这正是先验最该发挥作用的场景。
        known = (
            illuminant_guess in config.ILLUMINANT_GUESS_CCT
            and illuminant_guess != "unknown"
        )
        if known:
            weight = config.ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK
            blended = (1.0 - weight) * mired(D65_CCT_K) + weight * mired(
                config.ILLUMINANT_GUESS_CCT[illuminant_guess]
            )
            cct = from_mired(blended)
            locus = config.ILLUMINANT_GUESS_LOCUS.get(illuminant_guess)
            corrected = locus_xy_at_cct(cct, locus)
            return {
                "method": method,
                "cct": float(cct),
                "duv": 0.0,
                "xy": [float(corrected[0]), float(corrected[1])],
                "assumedD65": True,
                "locus": locus,
                "priorApplied": True,
                "priorWeight": float(weight),
            }
        return {
            "method": method,
            "cct": D65_CCT_K,
            "duv": 0.0,
            "xy": list(D65_XY),
            "assumedD65": True,
            "locus": None,
            "priorApplied": False,
            "priorWeight": 0.0,
        }

    locus_hint = config.ILLUMINANT_GUESS_LOCUS.get(illuminant_guess)
    cct_measured, duv, locus = cct_and_duv(xy_estimate, locus_hint)

    if prior_weight is None:
        prior_weight = (
            config.ILLUMINANT_PRIOR_WEIGHT if illuminant_guess != "unknown" else 0.0
        )
    prior_cct = config.ILLUMINANT_GUESS_CCT.get(illuminant_guess, D65_CCT_K)
    if prior_weight > 0.0:
        blended_mired = (1.0 - prior_weight) * mired(cct_measured) + prior_weight * mired(prior_cct)
        cct = from_mired(blended_mired)
    else:
        cct = cct_measured

    corrected_xy = locus_xy_at_cct(cct, locus)
    return {
        "method": method,
        "cct": float(cct),
        "duv": float(duv),
        "xy": [float(corrected_xy[0]), float(corrected_xy[1])],
        "assumedD65": bool(assumed_d65),
        "locus": locus,
        "priorApplied": bool(prior_weight > 0.0),
        "priorWeight": float(prior_weight),
    }

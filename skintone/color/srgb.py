"""sRGB <-> linear RGB <-> CIE XYZ <-> CIELAB, hand written (contract section 4).

Colour spaces
-------------
* ``srgb``   -- *encoded* sRGB, either 0..1 float or 0..255 integer, following
                the IEC 61966-2-1 transfer function.
* ``linear`` -- *linear* sRGB primaries in 0..1; the only space in which
                averaging, least squares and matrix work is allowed.
* ``xyz``    -- CIE 1931 2-degree tristimulus. White point is D65 unless the
                caller states otherwise.
* ``lab``    -- CIELAB with D65 white point ``(Xn, Yn, Zn) =
                (0.95047, 1.0, 1.08883)``.
* ``xy``/``xyY`` -- CIE 1931 chromaticity / luminance.
* ``lch``    -- CIELAB in cylindrical form; ``h`` in degrees in ``[0, 360)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Linear sRGB -> CIE XYZ, D65, 2-degree observer (IEC 61966-2-1 / Bruce Lindbloom).
XYZ_FROM_LINEAR_RGB: NDArray[np.float64] = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

#: Inverse of :data:`XYZ_FROM_LINEAR_RGB`.
LINEAR_RGB_FROM_XYZ: NDArray[np.float64] = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

#: D65 white point used by every CIELAB conversion in this package.
D65_XYZ: NDArray[np.float64] = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

#: D65 chromaticity coordinates (CIE 1931, 2-degree).
D65_XY: tuple[float, float] = (0.31271, 0.32902)

#: D65 correlated colour temperature in kelvin (nominal, for reporting only).
D65_CCT_K = 6504.0

_SRGB_LINEAR_CUTOFF = 0.04045
_SRGB_ENCODE_CUTOFF = 0.0031308
_SRGB_SLOPE = 12.92
_SRGB_ALPHA = 1.055
_SRGB_OFFSET = 0.055
_SRGB_GAMMA = 2.4
_SRGB_GAMMA_INV = 1.0 / 2.4

_LAB_EPS = 216.0 / 24389.0  # (6/29)^3
_LAB_KAPPA = 24389.0 / 27.0  # (29/3)^3
_LAB_DELTA = 6.0 / 29.0


def _as_array(value: ArrayLike) -> NDArray[np.float64]:
    """Return ``value`` as a float64 ndarray (0-d for scalars)."""
    return np.asarray(value, dtype=np.float64)


def srgb_to_linear(srgb: ArrayLike) -> NDArray[np.float64]:
    """Decode *encoded* sRGB (0..1) to linear sRGB (0..1).

    Implements IEC 61966-2-1 exactly::

        c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4

    Args:
        srgb: encoded sRGB values in 0..1 (any shape). Values outside 0..1 are
            passed through the same formula (they are already invalid input).

    Returns:
        Linear sRGB of the same shape, 0..1.
    """
    c = _as_array(srgb)
    return np.where(
        c <= _SRGB_LINEAR_CUTOFF,
        c / _SRGB_SLOPE,
        ((c + _SRGB_OFFSET) / _SRGB_ALPHA) ** _SRGB_GAMMA,
    )


def linear_to_srgb(linear: ArrayLike) -> NDArray[np.float64]:
    """Encode linear sRGB (0..1) back to *encoded* sRGB (0..1).

    Implements IEC 61966-2-1 exactly::

        l <= 0.0031308 ? 12.92 * l : 1.055 * l ** (1 / 2.4) - 0.055

    Args:
        linear: linear sRGB in 0..1 (any shape).

    Returns:
        Encoded sRGB of the same shape, 0..1.
    """
    value = _as_array(linear)
    safe = np.clip(value, 0.0, None)
    return np.where(
        value <= _SRGB_ENCODE_CUTOFF,
        value * _SRGB_SLOPE,
        _SRGB_ALPHA * safe**_SRGB_GAMMA_INV - _SRGB_OFFSET,
    )


def srgb8_to_linear(srgb8: ArrayLike) -> NDArray[np.float64]:
    """Decode 8-bit sRGB (0..255) to linear sRGB (0..1)."""
    return srgb_to_linear(_as_array(srgb8) / 255.0)


def linear_to_srgb8(linear: ArrayLike) -> NDArray[np.float64]:
    """Encode linear sRGB (0..1) to 8-bit sRGB (0..255), unrounded floats."""
    return linear_to_srgb(linear) * 255.0


def linear_rgb_to_xyz(linear_rgb: ArrayLike) -> NDArray[np.float64]:
    """Convert linear sRGB (0..1) to CIE XYZ (D65).

    Args:
        linear_rgb: array with a trailing axis of length 3, values 0..1.

    Returns:
        CIE XYZ, same leading shape, D65 white.
    """
    rgb = _as_array(linear_rgb)
    return rgb @ XYZ_FROM_LINEAR_RGB.T


def xyz_to_linear_rgb(xyz: ArrayLike) -> NDArray[np.float64]:
    """Convert CIE XYZ to linear sRGB (0..1). Out-of-gamut values are kept.

    Args:
        xyz: array with a trailing axis of length 3.

    Returns:
        Linear sRGB, same leading shape. Values may fall outside 0..1 when the
        input colour is outside the sRGB gamut; the caller decides whether to
        clip.
    """
    value = _as_array(xyz)
    return value @ LINEAR_RGB_FROM_XYZ.T


def xyz_to_xy(xyz: ArrayLike) -> NDArray[np.float64]:
    """Convert CIE XYZ to CIE 1931 chromaticity ``(x, y)``.

    Args:
        xyz: array with a trailing axis of length 3.

    Returns:
        Chromaticity ``(x, y)`` with the same leading shape. A black input
        (sum == 0) returns the D65 chromaticity instead of dividing by zero.
    """
    value = _as_array(xyz)
    total = np.sum(value, axis=-1, keepdims=True)
    safe = np.where(np.abs(total) < 1e-12, 1.0, total)
    xy = value[..., :2] / safe
    return np.where(np.abs(total) < 1e-12, np.array(D65_XY), xy)


def xy_to_xyz(xy: ArrayLike, luminance: float | ArrayLike = 1.0) -> NDArray[np.float64]:
    """Convert CIE 1931 chromaticity to CIE XYZ with a given luminance Y.

    Args:
        xy: chromaticity ``(x, y)``, trailing axis of length 2.
        luminance: Y value (1.0 -> the colour normalised to unit luminance).

    Returns:
        CIE XYZ array.
    """
    value = _as_array(xy)
    x, y = value[..., 0], value[..., 1]
    safe_y = np.where(np.abs(y) < 1e-12, 1e-12, y)
    y_lum = _as_array(luminance)
    return np.stack([x / safe_y * y_lum, y_lum * np.ones_like(x), (1.0 - x - y) / safe_y * y_lum], axis=-1)


def xyz_to_xyy(xyz: ArrayLike) -> NDArray[np.float64]:
    """Convert CIE XYZ to ``xyY`` (chromaticity plus luminance).

    Args:
        xyz: array with a trailing axis of length 3.

    Returns:
        ``[x, y, Y]``.
    """
    value = _as_array(xyz)
    xy = xyz_to_xy(value)
    return np.concatenate([xy, value[..., 1:2]], axis=-1)


def xyy_to_xyz(xyy: ArrayLike) -> NDArray[np.float64]:
    """Convert ``xyY`` back to CIE XYZ.

    Args:
        xyy: array with a trailing axis of length 3.

    Returns:
        CIE XYZ array.
    """
    value = _as_array(xyy)
    return xy_to_xyz(value[..., :2], value[..., 2])


def _lab_f(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """CIELAB non-linearity ``f(t)``."""
    return np.where(t > _LAB_EPS, np.cbrt(t), (_LAB_KAPPA * t + 16.0) / 116.0)


def _lab_f_inv(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse CIELAB non-linearity: ``t > 6/29 ? t**3 : ...``."""
    return np.where(t > _LAB_DELTA, t**3, (t - 16.0 / 116.0) / _LAB_KAPPA)


def xyz_to_lab(xyz: ArrayLike, white: ArrayLike = D65_XYZ) -> NDArray[np.float64]:
    """Convert CIE XYZ to CIELAB.

    Args:
        xyz: array with a trailing axis of length 3.
        white: reference white XYZ; default D65 ``(0.95047, 1.0, 1.08883)``.

    Returns:
        ``[L*, a*, b*]``; ``L*`` in 0..100, ``a*``/``b*`` roughly -128..127.
    """
    value = _as_array(xyz)
    wp = _as_array(white)
    f = _lab_f(value / wp)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )


def lab_to_xyz(lab: ArrayLike, white: ArrayLike = D65_XYZ) -> NDArray[np.float64]:
    """Convert CIELAB back to CIE XYZ.

    Args:
        lab: array with a trailing axis of length 3, ``L*`` in 0..100.
        white: reference white XYZ; default D65.

    Returns:
        CIE XYZ array.
    """
    value = _as_array(lab)
    wp = _as_array(white)
    fy = (value[..., 0] + 16.0) / 116.0
    fx = fy + value[..., 1] / 500.0
    fz = fy - value[..., 2] / 200.0
    return np.stack([_lab_f_inv(fx), _lab_f_inv(fy), _lab_f_inv(fz)], axis=-1) * wp


def linear_rgb_to_lab(linear_rgb: ArrayLike, white: ArrayLike = D65_XYZ) -> NDArray[np.float64]:
    """Convert linear sRGB (0..1, D65) to CIELAB (D65).

    Args:
        linear_rgb: array with a trailing axis of length 3, values 0..1.
        white: reference white; default D65.

    Returns:
        ``[L*, a*, b*]``.
    """
    return xyz_to_lab(linear_rgb_to_xyz(linear_rgb), white)


def lab_to_linear_rgb(lab: ArrayLike, white: ArrayLike = D65_XYZ) -> NDArray[np.float64]:
    """Convert CIELAB (D65) to linear sRGB (0..1, may be out of gamut)."""
    return xyz_to_linear_rgb(lab_to_xyz(lab, white))


def srgb8_to_lab(srgb8: ArrayLike, white: ArrayLike = D65_XYZ) -> NDArray[np.float64]:
    """Convert 8-bit sRGB (0..255) to CIELAB (D65)."""
    return linear_rgb_to_lab(srgb8_to_linear(srgb8), white)


def white_xyz_from_xy(xy: ArrayLike, luminance: float = 1.0) -> NDArray[np.float64]:
    """Return a white point as CIE XYZ from its CIE 1931 chromaticity.

    Args:
        xy: chromaticity ``(x, y)``.
        luminance: value of ``Y``.

    Returns:
        XYZ triple with ``Y == luminance``.
    """
    value = _as_array(xy).reshape(2)
    x, y = float(value[0]), float(value[1])
    if abs(y) < 1e-12:
        raise ValueError("degenerate chromaticity with y == 0")
    return np.array(
        [x / y * luminance, luminance, (1.0 - x - y) / y * luminance], dtype=np.float64
    )


def lab_to_lch(lab: ArrayLike) -> NDArray[np.float64]:
    """Convert CIELAB to cylindrical ``LCh``.

    Args:
        lab: array with a trailing axis of length 3.

    Returns:
        ``[L*, C*, h_ab]`` with ``C* = hypot(a*, b*)`` and ``h_ab`` in degrees
        in ``[0, 360)`` (0 degrees = +a* = red, 90 degrees = +b* = yellow).
    """
    value = _as_array(lab)
    a, b = value[..., 1], value[..., 2]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([value[..., 0], chroma, hue], axis=-1)


def lch_to_lab(lch: ArrayLike) -> NDArray[np.float64]:
    """Convert cylindrical ``LCh`` (h in degrees) back to CIELAB."""
    value = _as_array(lch)
    lightness, chroma, hue = value[..., 0], value[..., 1], value[..., 2]
    rad = np.radians(hue)
    return np.stack([lightness, chroma * np.cos(rad), chroma * np.sin(rad)], axis=-1)


def hex_to_srgb8(value: str) -> NDArray[np.float64]:
    """Parse ``"#rrggbb"`` (or ``"rrggbb"``) into 8-bit sRGB ``[r, g, b]``."""
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {value!r}")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def srgb8_to_hex(srgb8: ArrayLike) -> str:
    """Format 8-bit sRGB as ``"#rrggbb"``, clipping and rounding to int."""
    value = np.clip(_as_array(srgb8), 0.0, 255.0)
    r, g, b = (int(round(float(v))) for v in value.reshape(-1)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def linear_to_hex(linear_rgb: ArrayLike) -> str:
    """Format linear sRGB (0..1) as ``"#rrggbb"`` (gamut-clipped)."""
    return srgb8_to_hex(linear_to_srgb8(linear_rgb))

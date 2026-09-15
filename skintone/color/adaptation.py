"""Chromatic adaptation: Bradford and von Kries (contract section 4).

A chromatic adaptation transform (CAT) maps tristimulus values seen under a
source white to the corresponding values under a destination white, using a
per-cone gain in a cone-response space::

    M = LMS_from_XYZ ;  d = (M @ white_dst) / (M @ white_src)
    CAT = M^-1 @ diag(d) @ M

The default in this project is **Bradford**, chosen because it is the matrix
behind ICC v4 and every mainstream colour pipeline, so results can be compared
against other tools.

Colour spaces
-------------
* Inputs and outputs are CIE XYZ (any luminance scale) or linear sRGB (0..1);
  adaptation is a linear operation and must never be applied to encoded sRGB.
* White points are given as XYZ triples normalised to ``Y = 1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .srgb import D65_XYZ, LINEAR_RGB_FROM_XYZ, XYZ_FROM_LINEAR_RGB

#: Bradford cone-response matrix (linear sRGB-ish "sharpened" cones).
BRADFORD_LMS_FROM_XYZ: NDArray[np.float64] = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

#: Inverse of :data:`BRADFORD_LMS_FROM_XYZ`.
BRADFORD_XYZ_FROM_LMS: NDArray[np.float64] = np.array(
    [
        [0.9869929, -0.1470543, 0.1599627],
        [0.4323053, 0.5183603, 0.0492912],
        [-0.0085287, 0.0400428, 0.9684867],
    ],
    dtype=np.float64,
)

#: Hunt-Pointer-Estevez cone-response matrix, the classic "von Kries" basis.
VON_KRIES_LMS_FROM_XYZ: NDArray[np.float64] = np.array(
    [
        [0.40024, 0.70760, -0.08081],
        [-0.22630, 1.16532, 0.04570],
        [0.00000, 0.00000, 0.91822],
    ],
    dtype=np.float64,
)

#: Inverse of :data:`VON_KRIES_LMS_FROM_XYZ`.
VON_KRIES_XYZ_FROM_LMS: NDArray[np.float64] = np.array(
    [
        [1.8599364, -1.1293816, 0.2198974],
        [0.3611914, 0.6388125, -0.0000064],
        [0.0000000, 0.0000000, 1.0890636],
    ],
    dtype=np.float64,
)

#: Identity matrix in XYZ: "XYZ scaling", i.e. a pure diagonal von Kries gain.
XYZ_SCALING_LMS_FROM_XYZ: NDArray[np.float64] = np.eye(3, dtype=np.float64)

#: Supported adaptation method names -> cone-response basis.
CAT_BASES: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
    "Bradford": (BRADFORD_LMS_FROM_XYZ, BRADFORD_XYZ_FROM_LMS),
    "von Kries": (VON_KRIES_LMS_FROM_XYZ, VON_KRIES_XYZ_FROM_LMS),
    "XYZ scaling": (XYZ_SCALING_LMS_FROM_XYZ, XYZ_SCALING_LMS_FROM_XYZ),
}

DEFAULT_ADAPTATION = "Bradford"


def normalized_white(white: ArrayLike) -> NDArray[np.float64]:
    """Return a white point XYZ triple normalised to ``Y = 1``.

    Args:
        white: XYZ triple (any luminance scale, e.g. ``[95.05, 100, 108.88]``).

    Returns:
        XYZ triple with ``Y == 1``.
    """
    value = np.asarray(white, dtype=np.float64).reshape(3)
    if value[1] <= 0:
        raise ValueError("white point must have a positive Y component")
    return value / value[1]


def adaptation_matrix(
    src_white: ArrayLike,
    dst_white: ArrayLike,
    method: str = DEFAULT_ADAPTATION,
) -> NDArray[np.float64]:
    """Build the 3x3 chromatic adaptation matrix ``src_white -> dst_white``.

    Args:
        src_white: XYZ of the illuminant the data was captured under (Y=1 scale
            is applied internally).
        dst_white: XYZ of the target illuminant, normally D65.
        method: one of :data:`CAT_BASES` (``"Bradford"``, ``"von Kries"``,
            ``"XYZ scaling"``).

    Returns:
        A 3x3 matrix ``M`` such that ``xyz_dst = M @ xyz_src``.

    Raises:
        ValueError: if ``method`` is unknown or the white points are degenerate.
    """
    if method not in CAT_BASES:
        raise ValueError(
            f"unknown adaptation method {method!r}; expected one of {sorted(CAT_BASES)}"
        )
    lms_from_xyz, xyz_from_lms = CAT_BASES[method]
    src = normalized_white(src_white)
    dst = normalized_white(dst_white)

    src_lms = lms_from_xyz @ src
    dst_lms = lms_from_xyz @ dst
    if np.any(np.abs(src_lms) < 1e-12):
        raise ValueError("source white collapses in the cone-response basis")
    gains = dst_lms / src_lms
    return xyz_from_lms @ np.diag(gains) @ lms_from_xyz


def adapt_xyz(
    xyz: ArrayLike,
    src_white: ArrayLike,
    dst_white: ArrayLike = D65_XYZ,
    method: str = DEFAULT_ADAPTATION,
) -> NDArray[np.float64]:
    """Chromatic-adapt CIE XYZ values from one illuminant to another.

    Args:
        xyz: CIE XYZ, trailing axis of length 3 (relative units; the source
            white defines the scale).
        src_white: source illuminant XYZ.
        dst_white: destination illuminant XYZ; default D65.
        method: ``"Bradford"`` (default) or ``"von Kries"``.

    Returns:
        Adapted CIE XYZ, same shape. The result is relative to ``dst_white``.
    """
    matrix = adaptation_matrix(src_white, dst_white, method)
    return np.asarray(xyz, dtype=np.float64) @ matrix.T


def adapt_linear_rgb(
    linear_rgb: ArrayLike,
    src_white: ArrayLike,
    dst_white: ArrayLike = D65_XYZ,
    method: str = DEFAULT_ADAPTATION,
) -> NDArray[np.float64]:
    """Chromatic-adapt **linear** sRGB from one illuminant to another.

    Args:
        linear_rgb: linear sRGB in 0..1, trailing axis of length 3. Must *not*
            be encoded sRGB.
        src_white: source illuminant XYZ (e.g. the estimated scene white).
        dst_white: destination illuminant XYZ; default D65.
        method: ``"Bradford"`` (default) or ``"von Kries"``.

    Returns:
        Linear sRGB, same shape, expressed under ``dst_white``. Values may leave
        0..1 if the adaptation pushed a colour out of gamut.
    """
    xyz = np.asarray(linear_rgb, dtype=np.float64) @ XYZ_FROM_LINEAR_RGB.T
    adapted = adapt_xyz(xyz, src_white, dst_white, method)
    return adapted @ LINEAR_RGB_FROM_XYZ.T

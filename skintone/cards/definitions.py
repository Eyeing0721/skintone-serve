"""Reference card ``skintone-a4-v1``: geometry and patch values (contract 3).

This module is the *code-ified* contract section 3 and is the single source of
truth for both repositories: the web client renders ``card.html`` from
``GET /v1/card/skintone-a4-v1``, and this server samples the same numbers back out
of a photograph.

Units
-----
* ``center_mm`` / ``size_mm`` -- millimetres on the printed A4 sheet, measured
  from the top-left corner of the paper.
* ``nominal_srgb`` -- **encoded** sRGB, 0..255 integers, D65, as printed.
* Pixel helpers return integer pixel coordinates in the canonical 2480x3508
  rasterisation (300 DPI, ``CARD_PX_PER_MM`` pixels per millimetre).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .. import config

PatchKind = Literal["gray", "color", "skin", "olive"]


@dataclass(frozen=True)
class Patch:
    """One printable patch on the card.

    Attributes:
        id: Stable identifier, also used as the key in calibration responses.
        kind: ``gray`` (neutrality constraint for the CCM), ``color``, ``skin``
            or ``olive``.
        center_mm: Patch centre ``(x, y)`` in millimetres from the paper's
            top-left corner.
        size_mm: Patch side length in millimetres.
        nominal_srgb: The **encoded** sRGB value the patch is designed to be,
            0..255 per channel.
    """

    id: str
    kind: PatchKind
    center_mm: tuple[float, float]
    size_mm: float
    nominal_srgb: tuple[int, int, int]


@dataclass(frozen=True)
class Marker:
    """One ArUco marker of the card.

    Attributes:
        id: ArUco id inside :data:`skintone.config.ARUCO_DICTIONARY`.
        center_mm: Marker centre in millimetres from the paper's top-left corner.
        size_mm: Side length of the marker's black square, in millimetres.
    """

    id: int
    center_mm: tuple[float, float]
    size_mm: float


#: The gray ramp (contract section 3). ``K05`` is the extra dark anchor in the
#: lower-left blank area; ``W``/``K``/``G90b``/``G50b`` live in row 4 below and are
#: also neutral, so the CCM gets a strong neutrality constraint.
_GRAY_PATCHES: tuple[Patch, ...] = (
    Patch("G90", "gray", (45.0, 65.0), config.GRAY_PATCH_SIZE_MM, (245, 245, 245)),
    Patch("G70", "gray", (77.0, 65.0), config.GRAY_PATCH_SIZE_MM, (200, 200, 200)),
    Patch("G50", "gray", (109.0, 65.0), config.GRAY_PATCH_SIZE_MM, (160, 160, 160)),
    Patch("G30", "gray", (141.0, 65.0), config.GRAY_PATCH_SIZE_MM, (120, 120, 120)),
    Patch("G10", "gray", (173.0, 65.0), config.GRAY_PATCH_SIZE_MM, (75, 75, 75)),
    Patch("K05", "gray", (25.0, 215.0), config.GRAY_PATCH_SIZE_MM, (40, 40, 40)),
)


def _grid_patch(
    patch_id: str,
    kind: PatchKind,
    column: int,
    row: int,
    nominal_srgb: tuple[int, int, int],
) -> Patch:
    """Build a colour-grid patch from its column/row index (contract section 3)."""
    x_mm = config.COLOR_PATCH_COL_X_MM[column]
    y_mm = config.COLOR_PATCH_ROW_Y_MM[row]
    return Patch(patch_id, kind, (x_mm, y_mm), config.COLOR_PATCH_SIZE_MM, nominal_srgb)


#: 6 columns x 4 rows, row-major (contract section 3). The R/G/B/C/M/Y and
#: orange/brown blocks are kept deliberately: the CCM least squares needs patches
#: that span a large volume of colour space, otherwise the solution is
#: ill-conditioned and extrapolates badly into the skin region.
_COLOR_GRID: tuple[Patch, ...] = (
    _grid_patch("R", "color", 0, 0, (192, 57, 43)),
    _grid_patch("G", "color", 1, 0, (39, 174, 96)),
    _grid_patch("B", "color", 2, 0, (46, 134, 193)),
    _grid_patch("C", "color", 3, 0, (23, 162, 184)),
    _grid_patch("M", "color", 4, 0, (142, 68, 173)),
    _grid_patch("Y", "color", 5, 0, (241, 196, 15)),
    _grid_patch("SK1", "skin", 0, 1, (243, 213, 192)),
    _grid_patch("SK2", "skin", 1, 1, (232, 185, 143)),
    _grid_patch("SK3", "skin", 2, 1, (198, 134, 66)),
    _grid_patch("SK4", "skin", 3, 1, (141, 85, 36)),
    _grid_patch("SK5", "skin", 4, 1, (245, 208, 197)),
    _grid_patch("SK6", "skin", 5, 1, (217, 166, 160)),
    _grid_patch("SK7", "skin", 0, 2, (161, 102, 94)),
    _grid_patch("SK8", "skin", 1, 2, (107, 74, 58)),
    _grid_patch("OL1", "olive", 2, 2, (181, 166, 66)),
    _grid_patch("OL2", "olive", 3, 2, (125, 140, 74)),
    _grid_patch("OR1", "color", 4, 2, (211, 84, 0)),
    _grid_patch("BR1", "color", 5, 2, (110, 75, 42)),
    _grid_patch("W", "gray", 0, 3, (255, 255, 255)),
    _grid_patch("K", "gray", 1, 3, (0, 0, 0)),
    _grid_patch("G90b", "gray", 2, 3, (245, 245, 245)),
    _grid_patch("G50b", "gray", 3, 3, (160, 160, 160)),
    _grid_patch("P1", "color", 4, 3, (232, 160, 180)),
    _grid_patch("T1", "color", 5, 3, (47, 79, 111)),
)

#: All patches of the card, gray ramp first.
PATCHES: tuple[Patch, ...] = _GRAY_PATCHES + _COLOR_GRID

#: Patches whose nominal chroma is zero; they pin the neutral axis of the CCM.
NEUTRAL_PATCH_IDS: tuple[str, ...] = tuple(
    patch.id for patch in PATCHES if patch.kind == "gray"
)

#: ArUco markers (contract section 3), ordered by id.
MARKERS: tuple[Marker, ...] = tuple(
    Marker(marker_id, center_mm, config.ARUCO_MARKER_SIZE_MM)
    for marker_id, center_mm in zip(config.ARUCO_MARKER_IDS, config.ARUCO_MARKER_CENTERS_MM)
)

#: Card ids this server can serve and analyse.
CARD_IDS: tuple[str, ...] = (config.CARD_ID,)

#: Printing/setup instructions returned by ``GET /v1/card/{card_id}``.
INSTRUCTIONS: tuple[str, ...] = (
    "用哑光纸打印，不要缩放（打印对话框中选择 100% / 实际大小，关闭\"适应页面\"）",
    "打印后在 300 DPI 下量取，四个标记中心应落在规格 ±0.5 mm 内",
    "拍摄时把卡片平放在脸颊或下颌旁，与皮肤同一平面、同一光照，不要遮挡卡片",
    "避免卡片反光：不要用闪光灯正对，尽量避开直射高光",
    "让四个黑色标记框完整入镜，卡片不要超出画面",
)


def get_patch(patch_id: str) -> Patch | None:
    """Return a patch by id, or ``None`` when the id is unknown."""
    for patch in PATCHES:
        if patch.id == patch_id:
            return patch
    return None


def card_ids() -> list[str]:
    """Return the ids of every card this server knows."""
    return list(CARD_IDS)


def patch_center_px(patch: Patch) -> tuple[float, float]:
    """Return a patch centre in canonical card pixels.

    Args:
        patch: The patch definition (``center_mm`` in millimetres).

    Returns:
        ``(x, y)`` in canonical raster pixels (300 DPI, 2480x3508).
    """
    return (
        patch.center_mm[0] * config.CARD_PX_PER_MM,
        patch.center_mm[1] * config.CARD_PX_PER_MM,
    )


def patch_sample_bounds_px(patch: Patch) -> tuple[int, int, int, int]:
    """Return the inclusive-exclusive pixel box sampled for a patch.

    The box is the central ``config.PATCH_SAMPLE_FRACTION`` (60 %) of the patch,
    which keeps printed edge bleed and paper texture out of the statistic
    (contract section 5A step 2).

    Args:
        patch: The patch definition.

    Returns:
        ``(x0, y0, x1, y1)`` clipped to the canonical raster; ``x1 <= x0`` means
        the patch is too small to sample.
    """
    center_x, center_y = patch_center_px(patch)
    half = 0.5 * patch.size_mm * config.CARD_PX_PER_MM * config.PATCH_SAMPLE_FRACTION
    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x1 = int(round(center_x + half))
    y1 = int(round(center_y + half))
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, config.CARD_WIDTH_PX)
    y1 = min(y1, config.CARD_HEIGHT_PX)
    return x0, y0, x1, y1


def marker_corners_mm(marker: Marker) -> tuple[tuple[float, float], ...]:
    """Return a marker's four corners in millimetres, in ArUco corner order.

    Order is top-left, top-right, bottom-right, bottom-left *in the marker's own
    upright orientation*, which is exactly the order ``cv2.aruco.detectMarkers``
    returns, so image corners can be paired with these by index.

    Args:
        marker: The marker definition.

    Returns:
        Four ``(x_mm, y_mm)`` tuples.
    """
    half = marker.size_mm / 2.0
    cx, cy = marker.center_mm
    return (
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    )


def marker_corners_px(marker: Marker) -> list[list[float]]:
    """Return a marker's four corners in canonical card pixels."""
    return [[x * config.CARD_PX_PER_MM, y * config.CARD_PX_PER_MM] for x, y in marker_corners_mm(marker)]


def card_spec(card_id: str) -> dict[str, Any] | None:
    """Return the printable specification of a card, or ``None`` if unknown.

    Args:
        card_id: Card identifier, e.g. ``"skintone-a4-v1"``.

    Returns:
        A dict matching the contract's ``GET /v1/card/{card_id}`` payload, with
        all millimetre values, marker centres, patch centres, patch sizes and
        nominal sRGB values, or ``None`` for an unknown id.
    """
    if card_id not in CARD_IDS:
        return None
    return {
        "cardId": config.CARD_ID,
        "specVersion": config.SPEC_VERSION,
        "paper": {
            "name": config.PAPER_NAME,
            "widthMm": config.PAPER_WIDTH_MM,
            "heightMm": config.PAPER_HEIGHT_MM,
            "dpi": config.CARD_DPI,
        },
        "markers": {
            "dictionary": config.ARUCO_DICTIONARY,
            "sizeMm": config.ARUCO_MARKER_SIZE_MM,
            "ids": [marker.id for marker in MARKERS],
            "centersMm": [[marker.center_mm[0], marker.center_mm[1]] for marker in MARKERS],
        },
        "patches": [
            {
                "id": patch.id,
                "kind": patch.kind,
                "centerMm": [patch.center_mm[0], patch.center_mm[1]],
                "sizeMm": patch.size_mm,
                "nominalSrgb": list(patch.nominal_srgb),
            }
            for patch in PATCHES
        ],
        "instructions": list(INSTRUCTIONS),
    }

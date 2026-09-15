"""Reference-card detection, sampling and CCM calibration (contract section 5A).

Pipeline
--------
1. Locate the four ArUco markers (``DICT_4X4_50`` ids 0-3) with ``cv2.aruco``.
2. Build a homography from their sixteen corners to the canonical card
   rasterisation (2480x3508 px, i.e. A4 at 300 DPI) and warp the photo onto it.
3. Sample each patch from the **median** of its central 60 % region, in linear
   light.
4. Solve a colour-correction matrix by least squares, ``M = XYZ_ref @ pinv(RGB)``,
   in either 3x3 form or 3x4 form with an offset term.
5. Score the fit per patch with CIEDE2000; the mean feeds the ``ccm_delta_e``
   gate.

Colour spaces
-------------
* Card image: OpenCV ``BGR`` ``uint8``, **encoded** sRGB.
* ``measured_linear`` / ``measured_srgb8``: linear sRGB 0..1 / encoded sRGB
  0..255, both per patch.
* ``reference_xyz``: CIE XYZ (D65) of the patch's nominal sRGB value.
* The CCM maps **linear** sRGB to CIE XYZ (D65), so it also performs the
  illuminant correction for the fitted scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .. import config
from ..cards import definitions
from ..color.ciede2000 import ciede2000
from ..color.srgb import (
    linear_rgb_to_xyz,
    srgb8_to_linear,
    srgb8_to_lab,
    xyz_to_lab,
    xyz_to_xy,
)


@dataclass(frozen=True)
class PatchSample:
    """One measured patch on the warped card.

    Attributes:
        id: Patch id, e.g. ``"G90"``.
        kind: ``gray`` / ``color`` / ``skin`` / ``olive``.
        measured_srgb8: Per-channel median of the sampled pixels in **encoded**
            sRGB 0..255 (this is what a user would see on screen).
        measured_linear: The same median converted to **linear** sRGB 0..1.
        nominal_srgb: The patch's designed **encoded** sRGB value, 0..255.
        reference_xyz: CIE XYZ (D65) of the patch's nominal sRGB value.
        reference_lab: CIELAB (D65) of the nominal value.
        usable: False when the patch saturated and carries no colour information.
    """

    id: str
    kind: str
    measured_srgb8: NDArray[np.float64]
    measured_linear: NDArray[np.float64]
    nominal_srgb: NDArray[np.float64]
    reference_xyz: NDArray[np.float64]
    reference_lab: NDArray[np.float64]
    usable: bool


@dataclass(frozen=True)
class CcmResult:
    """A fitted colour-correction matrix and its residual.

    Attributes:
        matrix: 3x3 or 3x4 matrix mapping **linear** sRGB to CIE XYZ (D65).
        kind: ``"3x3"`` or ``"3x4"``.
        delta_e_mean: Mean per-patch CIEDE2000 residual (the ``ccm_delta_e`` gate).
        delta_e_max: Largest per-patch CIEDE2000 residual.
        per_patch: ``{patch_id: deltaE00}`` for every usable patch.
    """

    matrix: NDArray[np.float64]
    kind: str
    delta_e_mean: float
    delta_e_max: float
    per_patch: dict[str, float]


def _dictionary() -> "cv2.aruco.Dictionary":
    """Return the ArUco dictionary named in :mod:`skintone.config`."""
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_markers(image_bgr: NDArray[np.uint8]) -> dict[int, NDArray[np.float64]]:
    """Detect the card's ArUco markers.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.

    Returns:
        ``{marker_id: corners}`` where ``corners`` is a ``(4, 2)`` float array in
        original image pixels, ordered top-left, top-right, bottom-right,
        bottom-left in the marker's own upright orientation (the order
        ``cv2.aruco`` guarantees, which is why it can be paired with
        :func:`skintone.cards.definitions.marker_corners_px` by index).
    """
    dictionary = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image_bgr)
    else:  # pragma: no cover - legacy OpenCV fallback
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, dictionary)
    found: dict[int, NDArray[np.float64]] = {}
    if ids is None:
        return found
    for marker_corners, marker_id in zip(corners, ids.ravel()):
        found[int(marker_id)] = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    return found


def homography_to_canonical(
    markers: dict[int, NDArray[np.float64]],
) -> NDArray[np.float64] | None:
    """Build the homography that maps the photo onto the canonical card raster.

    Args:
        markers: Output of :func:`detect_markers`.

    Returns:
        A 3x3 matrix ``H`` such that ``warpPerspective(image, H, (2480, 3508))``
        yields the upright card, or ``None`` when not all four markers are present.
    """
    source: list[list[float]] = []
    target: list[list[float]] = []
    for marker in definitions.MARKERS:
        corners = markers.get(marker.id)
        if corners is None:
            return None
        source.extend(corners.tolist())
        target.extend(definitions.marker_corners_px(marker))
    if len(source) != 4 * 4:
        return None
    matrix, _ = cv2.findHomography(
        np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64), 0
    )
    return None if matrix is None else np.asarray(matrix, dtype=np.float64)


def warp_to_canonical(
    image_bgr: NDArray[np.uint8], homography: NDArray[np.float64]
) -> NDArray[np.uint8]:
    """Warp a photo into the canonical 2480x3508 card raster.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        homography: Output of :func:`homography_to_canonical`.

    Returns:
        ``(3508, 2480, 3)`` ``uint8`` image, the card seen straight on.
    """
    return cv2.warpPerspective(
        image_bgr,
        homography,
        (config.CARD_WIDTH_PX, config.CARD_HEIGHT_PX),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def sample_patch(
    card_bgr: NDArray[np.uint8], patch: definitions.Patch
) -> PatchSample:
    """Sample one patch from the canonical card image.

    The sample is the per-channel **median** over the central
    ``config.PATCH_SAMPLE_FRACTION`` (60 %) of the patch, taken in linear light.
    A median (not a mean) keeps print speckle, dust and a stray highlight from
    dragging the result.

    Args:
        card_bgr: ``(3508, 2480, 3)`` canonical card image in encoded sRGB.
        patch: Patch definition (geometry and nominal value).

    Returns:
        A :class:`PatchSample`; ``usable`` is False when the sample saturated.
    """
    x0, y0, x1, y1 = definitions.patch_sample_bounds_px(patch)
    if x1 - x0 < config.PATCH_MIN_SAMPLE_PX or y1 - y0 < config.PATCH_MIN_SAMPLE_PX:
        nominal = np.asarray(patch.nominal_srgb, dtype=np.float64)
        return PatchSample(
            id=patch.id,
            kind=patch.kind,
            measured_srgb8=nominal,
            measured_linear=srgb8_to_linear(nominal),
            nominal_srgb=nominal,
            reference_xyz=linear_rgb_to_xyz(srgb8_to_linear(nominal)),
            reference_lab=srgb8_to_lab(nominal),
            usable=False,
        )

    region = card_bgr[y0:y1, x0:x1]
    pixels = np.ascontiguousarray(region.reshape(-1, 3)[:, ::-1]).astype(np.float64)
    srgb8_median = np.median(pixels, axis=0)
    linear_median = np.median(srgb8_to_linear(pixels.astype(np.uint8)), axis=0)

    nominal = np.asarray(patch.nominal_srgb, dtype=np.float64)
    usable = bool(np.all(srgb8_median < config.CCM_DROP_SATURATED_SRGB8))
    return PatchSample(
        id=patch.id,
        kind=patch.kind,
        measured_srgb8=srgb8_median,
        measured_linear=linear_median,
        nominal_srgb=nominal,
        reference_xyz=linear_rgb_to_xyz(srgb8_to_linear(nominal)),
        reference_lab=srgb8_to_lab(nominal),
        usable=usable,
    )


def sample_all_patches(card_bgr: NDArray[np.uint8]) -> list[PatchSample]:
    """Sample every patch defined by the card, in :data:`definitions.PATCHES` order."""
    return [sample_patch(card_bgr, patch) for patch in definitions.PATCHES]


def solve_ccm(
    measured_linear: NDArray[np.float64],
    reference_xyz: NDArray[np.float64],
    kind: str = "3x3",
) -> NDArray[np.float64]:
    """Solve a colour-correction matrix by least squares.

    ``M = XYZ_ref @ pinv(RGB_meas)`` in the contract's notation, i.e. the
    minimum-norm least-squares solution of ``XYZ_ref ~= M @ RGB_meas``. For the
    3x4 form the measured data is augmented with a constant 1 row, which gives the
    fit a black-level offset (used by cameras that lift black).

    Args:
        measured_linear: ``(N, 3)`` measured **linear** sRGB in 0..1.
        reference_xyz: ``(N, 3)`` reference CIE XYZ (D65) for the same patches.
        kind: ``"3x3"`` or ``"3x4"``.

    Returns:
        ``(3, 3)`` or ``(3, 4)`` float matrix mapping linear sRGB (plus an
        optional constant) to CIE XYZ (D65).

    Raises:
        ValueError: if ``kind`` is neither ``"3x3"`` nor ``"3x4"``, the inputs
            disagree in length, or there are too few patches.
    """
    measured = np.asarray(measured_linear, dtype=np.float64)
    reference = np.asarray(reference_xyz, dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1] != 3:
        raise ValueError("measured_linear must be (N, 3)")
    if reference.shape != measured.shape:
        raise ValueError("reference_xyz must have the same shape as measured_linear")
    rows = 4 if kind == "3x4" else 3
    if kind not in ("3x3", "3x4"):
        raise ValueError(f"unknown CCM kind {kind!r}")
    if measured.shape[0] < rows + 1:
        raise ValueError("not enough patches to solve a CCM")

    design = measured.T
    if kind == "3x4":
        design = np.vstack([design, np.ones((1, design.shape[1]), dtype=np.float64)])
    return reference.T @ np.linalg.pinv(design)


def apply_ccm(linear_rgb: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a CCM to **linear** sRGB values.

    Args:
        linear_rgb: ``(..., 3)`` linear sRGB in 0..1.
        matrix: ``(3, 3)`` or ``(3, 4)`` matrix from :func:`solve_ccm`.

    Returns:
        CIE XYZ (D65), same leading shape as ``linear_rgb``.
    """
    value = np.asarray(linear_rgb, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        value = np.concatenate(
            [value, np.ones(value.shape[:-1] + (1,), dtype=np.float64)], axis=-1
        )
    return value @ matrix.T


def choose_ccm(samples: list[PatchSample]) -> CcmResult:
    """Fit both CCM forms to the usable patches and pick one.

    Selection rule (documented, and driven by a constant): the 3x4 offset model is
    used only when the black reference patch (nominal ``#000000``) measures a
    lifted black level above :data:`skintone.config.CCM_OFFSET_BLACK_LEVEL`;
    otherwise the smaller, better-conditioned 3x3 model wins.

    Args:
        samples: Patch samples from :func:`sample_all_patches`.

    Returns:
        A :class:`CcmResult` with the per-patch CIEDE2000 residuals of the chosen
        model.

    Raises:
        ValueError: if fewer than :data:`skintone.config.CCM_MIN_PATCHES` patches
            are usable.
    """
    usable = [sample for sample in samples if sample.usable]
    if len(usable) < config.CCM_MIN_PATCHES:
        raise ValueError(
            f"only {len(usable)} usable patches, need {config.CCM_MIN_PATCHES}"
        )
    measured = np.stack([sample.measured_linear for sample in usable])
    reference = np.stack([sample.reference_xyz for sample in usable])

    black = next((s for s in usable if s.id == "K"), None)
    black_level = float(np.mean(black.measured_linear)) if black is not None else 0.0
    kind = "3x4" if black_level > config.CCM_OFFSET_BLACK_LEVEL else "3x3"

    matrix = solve_ccm(measured, reference, kind)
    corrected = apply_ccm(measured, matrix)
    corrected_lab = xyz_to_lab(corrected)
    reference_lab = np.stack([sample.reference_lab for sample in usable])
    differences = ciede2000(reference_lab, corrected_lab)
    per_patch = {
        sample.id: float(difference) for sample, difference in zip(usable, differences)
    }
    return CcmResult(
        matrix=matrix,
        kind=kind,
        delta_e_mean=float(np.mean(differences)),
        delta_e_max=float(np.max(differences)),
        per_patch=per_patch,
    )


def card_white_xy(samples: list[PatchSample]) -> NDArray[np.float64] | None:
    """Estimate the scene illuminant chromaticity from the card's neutral patches.

    The gray ramp and the neutral row-4 blocks are printed neutral, so whatever
    chromaticity they measure is the illuminant (times the camera's own response).
    Patches that saturated are ignored, and the black patches are down-weighted by
    excluding anything below 5 % linear, where sensor noise dominates.

    Args:
        samples: Patch samples from :func:`sample_all_patches`.

    Returns:
        CIE 1931 ``(x, y)`` of the estimated illuminant, or ``None`` when no
        neutral patch is usable.
    """
    neutrals = [
        sample
        for sample in samples
        if sample.kind == "gray"
        and sample.usable
        and float(np.mean(sample.measured_linear)) > 0.05
    ]
    if not neutrals:
        return None
    mean_linear = np.mean(np.stack([sample.measured_linear for sample in neutrals]), axis=0)
    if float(np.sum(mean_linear)) <= 0.0:
        return None
    return np.asarray(xyz_to_xy(linear_rgb_to_xyz(mean_linear)), dtype=np.float64)

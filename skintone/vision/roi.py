"""Polygon ROI -> mask -> robust skin statistics (contract sections 2.3 and 5).

The ROI stage is where most "wrong skin tone" results come from, so it is
deliberately conservative:

1. Build a mask from the **original-image pixel** polygons the client sends.
2. Reject pixels that carry no colour information: any channel at 250 or above
   (saturated) or at 5 or below (crushed) in encoded sRGB. These are the gate
   ``clipping`` values, measured over the whole ROI.
3. Reject specular pixels, defined as CIELAB ``L* > 85``. This is the gate
   ``specular``.
4. Reduce the survivors with robust statistics -- a per-channel **median** in
   linear light, plus a trimmed mean for diagnostics. Never a plain mean, and
   never in encoded sRGB (contract section 4).

Colour spaces
-------------
* Input images are OpenCV ``BGR`` ``uint8`` (encoded sRGB).
* All statistics are computed on **linear** sRGB in 0..1; the median/trimmed mean
  are taken there and only then converted to CIELAB (D65).
* ``dispersion`` is in CIEDE2000 units: the median per-pixel colour difference to
  the ROI's own median colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .. import config
from ..color.ciede2000 import ciede2000
from ..color.srgb import linear_rgb_to_lab, srgb8_to_linear


@dataclass(frozen=True)
class RoiStats:
    """Everything the gates and the result need from one skin ROI.

    Attributes:
        roi_pixels: Number of pixels inside the polygons, before rejection.
        valid_pixels: Number of usable pixels after clipping/specular rejection.
        clipping_ratio: Fraction of ROI pixels with any channel >= 250 or <= 5.
        specular_ratio: Fraction of ROI pixels with CIELAB ``L* > 85``.
        median_linear_rgb: Per-channel **median** of the valid pixels, linear
            sRGB in 0..1 (D65). This is the estimator used for the result.
        trimmed_mean_linear_rgb: Per-channel trimmed mean of the valid pixels,
            linear sRGB in 0..1, with ``ROI_TRIM_FRACTION`` cut from each tail.
        lab: CIELAB (D65) of ``median_linear_rgb``.
        dispersion: Median per-pixel CIEDE2000 distance to ``lab``.
        regions: Labels of the polygons that contributed, in input order.
    """

    roi_pixels: int
    valid_pixels: int
    clipping_ratio: float
    specular_ratio: float
    median_linear_rgb: NDArray[np.float64] | None
    trimmed_mean_linear_rgb: NDArray[np.float64] | None
    lab: NDArray[np.float64] | None
    dispersion: float | None
    regions: list[str]


def polygons_to_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    polygons: list[list[list[float]]],
) -> NDArray[np.uint8]:
    """Rasterise closed polygons into a binary mask.

    Args:
        shape: Image shape; only the first two entries (height, width) are used.
        polygons: Each polygon is a list of ``[x, y]`` points in **original image
            pixels**, closing implicitly. Fewer than three points are ignored.

    Returns:
        ``uint8`` mask of shape ``(height, width)`` with 255 inside the union of
        the polygons and 0 outside.
    """
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for points in polygons:
        if len(points) < 3:
            continue
        contour = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        contour = np.round(contour).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
    return mask


def image_bgr_to_linear_rgb(image_bgr: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Convert an OpenCV BGR ``uint8`` image to linear sRGB in 0..1.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in **encoded** sRGB (BGR order).

    Returns:
        ``(H, W, 3)`` float64 linear sRGB in 0..1, channels in RGB order.
    """
    rgb8 = np.ascontiguousarray(image_bgr[:, :, ::-1])
    return srgb8_to_linear(rgb8)


def _trimmed_mean(values: NDArray[np.float64], fraction: float) -> NDArray[np.float64]:
    """Per-column trimmed mean of a ``(N, C)`` array.

    Args:
        values: ``(N, C)`` samples.
        fraction: Fraction cut from each tail (0.10 -> central 80 % kept).

    Returns:
        Length-``C`` array of means; returns the plain mean when trimming would
        leave nothing.
    """
    count = values.shape[0]
    cut = int(np.floor(count * fraction))
    if count - 2 * cut < 1:
        return np.mean(values, axis=0)
    ordered = np.sort(values, axis=0)
    return np.mean(ordered[cut : count - cut], axis=0)


def compute_roi_stats(
    image_bgr: NDArray[np.uint8],
    polygons: list[list[list[float]]],
    labels: list[str],
) -> RoiStats:
    """Measure one skin ROI: rejection ratios plus robust linear-light statistics.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image, **encoded** sRGB, BGR order.
        polygons: Skin polygons in original-image pixel coordinates.
        labels: Region labels parallel to ``polygons`` (e.g. ``["jaw", "neck"]``).

    Returns:
        A :class:`RoiStats`. When the mask is empty or every pixel is rejected,
        ``median_linear_rgb``/``lab``/``dispersion`` are ``None`` and the counts
        and ratios still describe what happened.
    """
    mask = polygons_to_mask(image_bgr.shape, polygons)
    roi_pixels = int(np.count_nonzero(mask))
    if roi_pixels == 0:
        return RoiStats(0, 0, 0.0, 0.0, None, None, None, None, list(labels))

    bgr_pixels = image_bgr[mask > 0]
    rgb8 = np.ascontiguousarray(bgr_pixels[:, ::-1]).astype(np.int16)
    clipped = np.any(rgb8 >= config.CLIP_HIGH_SRGB8, axis=1) | np.any(
        rgb8 <= config.CLIP_LOW_SRGB8, axis=1
    )
    linear = srgb8_to_linear(np.clip(rgb8, 0, 255).astype(np.uint8))
    lab = linear_rgb_to_lab(linear)
    specular = lab[:, 0] > config.SPECULAR_L_MAX
    valid = ~(clipped | specular)

    clipping_ratio = float(np.count_nonzero(clipped)) / roi_pixels
    specular_ratio = float(np.count_nonzero(specular)) / roi_pixels
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        return RoiStats(
            roi_pixels,
            0,
            clipping_ratio,
            specular_ratio,
            None,
            None,
            None,
            None,
            list(labels),
        )

    valid_linear = linear[valid]
    valid_lab = lab[valid]
    median_linear = np.median(valid_linear, axis=0)
    trimmed_linear = _trimmed_mean(valid_linear, config.ROI_TRIM_FRACTION)
    median_lab = linear_rgb_to_lab(median_linear)
    differences = ciede2000(valid_lab, median_lab)
    dispersion = float(np.median(differences))

    return RoiStats(
        roi_pixels=roi_pixels,
        valid_pixels=valid_pixels,
        clipping_ratio=clipping_ratio,
        specular_ratio=specular_ratio,
        median_linear_rgb=median_linear,
        trimmed_mean_linear_rgb=trimmed_linear,
        lab=median_lab,
        dispersion=dispersion,
        regions=list(labels),
    )


def mask_fraction(mask: NDArray[np.uint8]) -> float:
    """Return the fraction of a binary mask that is set (0.0 when empty)."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def neutral_candidates(
    image_bgr: NDArray[np.uint8],
    exclude_mask: NDArray[np.uint8],
) -> NDArray[np.float64]:
    """Collect near-neutral background pixels for no-card illuminant estimation.

    Pixels are selected from a border band (to bias towards background rather than
    the face) and kept only when they are not too dark. Within that band the
    *least saturated* ``BACKGROUND_NEUTRAL_QUANTILE`` of the pixels is kept, and
    those must also stay below ``BACKGROUND_MAX_SATURATION``. The relative pick
    matters: a genuinely neutral wall photographed under tungsten is chromatic, so
    an absolute saturation cut alone would reject every warm indoor capture. This
    is the "semantic mask" the contract requires -- a global gray-world average is
    explicitly forbidden, because on a face close-up skin dominates and gray-world
    would neutralise the very signal being measured.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image, **encoded** sRGB, BGR order.
        exclude_mask: ``uint8`` mask (255 = exclude) covering skin/card/face.

    Returns:
        ``(N, 3)`` array of **linear** sRGB 0..1 candidates, empty when none pass.
    """
    height, width = image_bgr.shape[:2]
    band = np.zeros((height, width), dtype=np.uint8)
    margin_y = int(round(height * config.BACKGROUND_EDGE_MARGIN_RATIO))
    margin_x = int(round(width * config.BACKGROUND_EDGE_MARGIN_RATIO))
    band[:margin_y, :] = 255
    band[height - margin_y :, :] = 255
    band[:, :margin_x] = 255
    band[:, width - margin_x :] = 255

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    value = hsv[:, :, 2]
    eligible = (band > 0) & (exclude_mask == 0) & (value >= config.BACKGROUND_MIN_VALUE)
    if not np.any(eligible):
        return np.zeros((0, 3), dtype=np.float64)

    flat_eligible = np.flatnonzero(eligible.reshape(-1))
    flat_saturation = saturation.reshape(-1)
    keep_count = max(
        int(round(flat_eligible.size * config.BACKGROUND_NEUTRAL_QUANTILE)),
        config.ILLUMINANT_MIN_CANDIDATE_PIXELS,
    )
    order = flat_eligible[np.argsort(flat_saturation[flat_eligible], kind="stable")]
    selected = order[: min(keep_count, order.size)]
    selected = selected[flat_saturation[selected] <= config.BACKGROUND_MAX_SATURATION]
    if selected.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pixels = image_bgr.reshape(-1, 3)[selected][:, ::-1]
    return srgb8_to_linear(np.ascontiguousarray(pixels))

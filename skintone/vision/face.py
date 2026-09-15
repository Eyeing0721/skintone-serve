"""Haar-cascade face/eye fallbacks and semantic illuminant candidates.

The cascades ship inside ``opencv-python-headless`` (``cv2.data.haarcascades``),
so nothing is downloaded at runtime.

Two jobs live here:

* **Auto ROI** -- when the client sends no polygons, derive conservative jaw and
  neck polygons from a detected face box.
* **Semantic illuminant candidates** -- collect *achromatic* pixels from the
  sclera, the teeth and the background, always excluding skin and card. The
  contract forbids a global gray-world estimate; the selection here is what makes
  the illuminant estimate meaningful on a face close-up.

Colour spaces
-------------
* Inputs are OpenCV ``BGR`` ``uint8`` (encoded sRGB); ``(x, y, w, h)`` boxes are
  in original image pixels.
* Returned candidate pixels are **linear** sRGB in 0..1, RGB channel order.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .. import config
from ..color.srgb import srgb8_to_linear

Box = tuple[int, int, int, int]


@lru_cache(maxsize=8)
def _cascade(name: str) -> cv2.CascadeClassifier:
    """Load and cache a Haar cascade shipped with OpenCV.

    Args:
        name: Cascade file name, e.g. ``"haarcascade_frontalface_default.xml"``.

    Returns:
        The loaded :class:`cv2.CascadeClassifier` (possibly empty if the file is
        missing, which makes ``detectMultiScale`` return an empty result rather
        than raising).
    """
    path = Path(cv2.data.haarcascades) / name
    return cv2.CascadeClassifier(str(path))


def _gray(image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return an equalised grayscale copy suitable for Haar detection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def detect_faces(image_bgr: NDArray[np.uint8]) -> list[Box]:
    """Detect frontal faces with the bundled Haar cascade.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.

    Returns:
        Face boxes ``(x, y, w, h)`` in original image pixels, largest first.
    """
    gray = _gray(image_bgr)
    minimum = int(round(min(image_bgr.shape[:2]) * config.FACE_DETECT_MIN_SIZE_RATIO))
    faces = _cascade(config.HAAR_FACE_CASCADE).detectMultiScale(
        gray,
        scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_NEIGHBORS,
        minSize=(max(minimum, 24), max(minimum, 24)),
    )
    boxes = [tuple(int(v) for v in box) for box in faces]
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes


#: Local alias so the cascade call reads clearly above.
FACE_NEIGHBORS = config.FACE_DETECT_MIN_NEIGHBORS


def detect_eyes(image_bgr: NDArray[np.uint8], face: Box) -> list[Box]:
    """Detect eyes inside a face box, returned in original image coordinates.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        face: Face box ``(x, y, w, h)`` in original image pixels.

    Returns:
        Eye boxes ``(x, y, w, h)`` in original image pixels, at most two.
    """
    x, y, w, h = face
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, image_bgr.shape[1]), min(y + h, image_bgr.shape[0])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    upper = image_bgr[y0 : y0 + int(round((y1 - y0) * 0.6)), x0:x1]
    if upper.size == 0:
        return []
    gray = _gray(upper)
    minimum = max(int(round((x1 - x0) * 0.10)), 8)
    eyes = _cascade(config.HAAR_EYE_CASCADE).detectMultiScale(
        gray,
        scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_NEIGHBORS,
        minSize=(minimum, minimum),
    )
    boxes = [(x0 + int(ex), y0 + int(ey), int(ew), int(eh)) for ex, ey, ew, eh in eyes]
    boxes.sort(key=lambda box: box[0])
    return boxes[:2]


def _face_relative_box(face: Box, x_range: tuple[float, float], y_range: tuple[float, float]) -> Box:
    """Build a box from face-relative fractions."""
    x, y, w, h = face
    x0 = int(round(x + w * x_range[0]))
    x1 = int(round(x + w * x_range[1]))
    y0 = int(round(y + h * y_range[0]))
    y1 = int(round(y + h * y_range[1]))
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def auto_skin_polygons(image_bgr: NDArray[np.uint8], face: Box) -> list[dict[str, object]]:
    """Derive conservative jaw and neck polygons from a face box.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image (used only for bounds clamping).
        face: Face box ``(x, y, w, h)`` in original image pixels.

    Returns:
        Two ROI dictionaries ``{"label": str, "points": [[x, y], ...]}`` in
        original image pixels: ``jaw`` (cheeks/lower face) and ``neck`` (directly
        below the chin).
    """
    height, width = image_bgr.shape[:2]
    jaw = _face_relative_box(face, config.AUTO_ROI_JAW_X, config.AUTO_ROI_JAW_Y)
    neck = _face_relative_box(face, config.AUTO_ROI_NECK_X, config.AUTO_ROI_NECK_Y)
    polygons: list[dict[str, object]] = []
    for label, box in (("jaw", jaw), ("neck", neck)):
        x, y, w, h = box
        x0 = float(np.clip(x, 0, width - 1))
        x1 = float(np.clip(x + w, 0, width - 1))
        y0 = float(np.clip(y, 0, height - 1))
        y1 = float(np.clip(y + h, 0, height - 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        polygons.append(
            {
                "label": label,
                "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            }
        )
    return polygons


def face_mask(image_bgr: NDArray[np.uint8], faces: list[Box]) -> NDArray[np.uint8]:
    """Return a mask (255 = exclude) covering the given face boxes.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image.
        faces: Face boxes in original image pixels.

    Returns:
        ``uint8`` mask, dilated slightly so that chin/jaw skin just outside the
        detector's box is excluded from illuminant candidates too.
    """
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for x, y, w, h in faces:
        pad = int(round(0.15 * h))
        cv2.rectangle(
            mask,
            (max(x - pad, 0), max(y - pad, 0)),
            (min(x + w + pad, image_bgr.shape[1] - 1), min(y + h + 2 * pad, image_bgr.shape[0] - 1)),
            255,
            -1,
        )
    return mask


def _bright_low_saturation(
    image_bgr: NDArray[np.uint8],
    box: Box,
    max_saturation: float,
    min_value: int,
    limit: int,
    *,
    rank: str = "saturation",
    quantile: float = 1.0,
) -> NDArray[np.float64]:
    """Select low-saturation bright pixels inside a box, in linear sRGB.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        box: ``(x, y, w, h)`` region of interest in original image pixels.
        max_saturation: Upper bound on OpenCV HSV saturation (0..1 scale).
        min_value: Lower bound on OpenCV HSV value (0..255).
        limit: Maximum number of pixels returned.
        rank: ``"saturation"`` keeps the *least saturated* pixels (the contract's
            rule for sclera); ``"brightness"`` keeps the *brightest* ones (the
            rule for teeth, which are the brightest desaturated thing in a mouth).
        quantile: Fraction of the qualifying pixels to keep, applied before
            ``limit``. Used with ``rank="brightness"`` so that a large grey
            background inside the box cannot outvote a small bright mouth.

    Returns:
        ``(N, 3)`` linear sRGB in 0..1, RGB order; empty when nothing qualifies.
    """
    x, y, w, h = box
    height, width = image_bgr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros((0, 3), dtype=np.float64)
    patch = image_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    value = hsv[:, :, 2].astype(np.float64)
    keep = (saturation <= max_saturation) & (hsv[:, :, 2] >= min_value)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)

    flat_patch = patch.reshape(-1, 3)
    indices = np.flatnonzero(keep.reshape(-1))
    key = saturation.reshape(-1)[indices] if rank == "saturation" else -value.reshape(-1)[indices]
    order = indices[np.argsort(key, kind="stable")]
    if quantile < 1.0:
        count = max(
            int(round(order.size * quantile)),
            min(config.ILLUMINANT_MIN_CANDIDATE_PIXELS, order.size),
        )
    else:
        count = order.size
    selected = flat_patch[order[: min(count, limit)]][:, ::-1]
    return srgb8_to_linear(np.ascontiguousarray(selected))


def sclera_candidates(image_bgr: NDArray[np.uint8], faces: list[Box]) -> NDArray[np.float64]:
    """Collect sclera ("eye white") pixels as illuminant candidates.

    Within each detected eye box, the least saturated pixels are kept -- sclera is
    the most achromatic surface available on a face, and ``illuminant`` is read
    from the *median* of those pixels (contract section 5B).

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        faces: Face boxes in original image pixels.

    Returns:
        ``(N, 3)`` linear sRGB in 0..1; empty when no eye (or no bright
        low-saturation pixel) is found.
    """
    collected: list[NDArray[np.float64]] = []
    total = 0
    for face in faces:
        for eye in detect_eyes(image_bgr, face):
            x, y, w, h = eye
            scale = config.SCLERA_EYE_PATCH_SCALE
            inset = (
                int(round(x + w * (0.5 - scale / 2.0))),
                int(round(y + h * (0.5 - scale / 2.0))),
                max(int(round(w * scale)), 2),
                max(int(round(h * scale)), 2),
            )
            pixels = _bright_low_saturation(
                image_bgr,
                inset,
                config.SCLERA_MAX_SATURATION,
                config.SCLERA_MIN_VALUE,
                config.SCLERA_MAX_PIXELS,
            )
            if pixels.size:
                collected.append(pixels)
                total += pixels.shape[0]
            if total >= config.SCLERA_MAX_PIXELS:
                break
    if not collected:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(collected, axis=0)[: config.SCLERA_MAX_PIXELS]


def teeth_candidates(image_bgr: NDArray[np.uint8], faces: list[Box]) -> NDArray[np.float64]:
    """Collect tooth pixels as illuminant candidates.

    A smile cascade locates the mouth when possible; otherwise the lower-centre
    band of the face box is used. Only bright, low-saturation pixels are kept.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        faces: Face boxes in original image pixels.

    Returns:
        ``(N, 3)`` linear sRGB in 0..1; empty when nothing qualifies (a closed
        mouth is the common case, and the caller must cope with that).
    """
    collected: list[NDArray[np.float64]] = []
    for face in faces:
        x, y, w, h = face
        mouth = _face_relative_box(face, config.AUTO_ROI_MOUTH_X, config.AUTO_ROI_MOUTH_Y)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, image_bgr.shape[1]), min(y + h, image_bgr.shape[0])
        lower = image_bgr[y0:y1, x0:x1]
        if lower.size:
            smiles = _cascade(config.HAAR_SMILE_CASCADE).detectMultiScale(
                _gray(lower),
                scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
                minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
                minSize=(max((x1 - x0) // 8, 12), max((y1 - y0) // 8, 6)),
            )
            if len(smiles):
                sx, sy, sw, sh = max(smiles, key=lambda box: box[2] * box[3])
                mouth = (x0 + int(sx), y0 + int(sy), int(sw), int(sh))
        pixels = _bright_low_saturation(
            image_bgr,
            mouth,
            config.TEETH_MAX_SATURATION,
            config.TEETH_MIN_VALUE,
            config.TEETH_MAX_PIXELS,
            rank="brightness",
            quantile=config.TEETH_BRIGHT_QUANTILE,
        )
        if pixels.size:
            collected.append(pixels)
    if not collected:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(collected, axis=0)[: config.TEETH_MAX_PIXELS]

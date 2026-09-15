"""Request orchestration: image in, contract-shaped result out.

This module owns the three routes of contract section 5 and the confidence
fusion of section 5.5. The HTTP layer (:mod:`skintone.api`) only parses multipart
input, calls :func:`analyze`, and serialises the result.

Colour-space pipeline
---------------------
* Decode to OpenCV ``BGR`` ``uint8``, **encoded** sRGB, EXIF orientation applied.
* Card route: warp to the canonical raster, median-sample patches, fit a CCM that
  maps **linear** sRGB to CIE XYZ (D65), then apply it to the skin ROI.
* No-card route: estimate one illuminant chromaticity from semantically masked
  achromatic surfaces, project it onto the Planckian/daylight locus, then
  Bradford-adapt the skin from that white to D65.
* Either way, the reported ``labD65``, ``linearRgb`` and ``hex`` are all in the
  D65 frame, so they are mutually consistent.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from . import config, schemas
from .cards import definitions
from .color import palette as palette_rules
from .color.adaptation import DEFAULT_ADAPTATION, adapt_xyz
from .color.ciede2000 import ciede2000
from .color.illuminant import solve_illuminant, white_xyz_from_xy
from .color.skin import (
    depth_class,
    hue_angle_deg,
    ita_degrees,
    melanin_hemoglobin,
    undertone,
)
from .color.srgb import (
    D65_XYZ,
    linear_rgb_to_xyz,
    linear_to_hex,
    linear_to_srgb8,
    srgb8_to_linear,
    xyz_to_lab,
    xyz_to_linear_rgb,
    xyz_to_xy,
)
from .vision import card as card_vision
from .vision import face as face_vision
from .vision import roi as roi_vision
from .vision.roi import RoiStats

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


@dataclass(frozen=True)
class GateSpec:
    """Declaration of one confidence gate.

    Attributes:
        id: Gate id from the contract table.
        threshold: Comparison threshold.
        maximum: True for a "value must be at most threshold" gate.
        unit: Suffix used when rendering the measured value.
        label: Human-readable name used in the gate message.
    """

    id: str
    threshold: float
    maximum: bool
    unit: str
    label: str


#: Gate declarations, in the order they appear in the response.
GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("ccm_delta_e", config.CCM_DELTA_E_MAX, True, "", "色卡标定残差 ΔE00 均值"),
    GateSpec("clipping", config.CLIP_RATIO_MAX, True, "", "截断像素占比"),
    GateSpec("specular", config.SPECULAR_RATIO_MAX, True, "", "高光像素占比"),
    GateSpec("roi_area", float(config.ROI_MIN_PIXELS), False, "", "有效皮肤像素"),
    GateSpec("roi_dispersion", config.ROI_DISPERSION_MAX, True, "", "ROI 内 Lab 色差离散度"),
    GateSpec(
        "illuminant_residual",
        config.ILLUMINANT_DUV_MAX,
        True,
        "",
        "光源投影残差 |Duv|",
    ),
)

#: Gate ids that only apply to a given mode.
_GATES_BY_MODE: dict[str, tuple[str, ...]] = {
    "card": ("ccm_delta_e", "clipping", "specular", "roi_area", "roi_dispersion"),
    "nocard": ("clipping", "specular", "roi_area", "roi_dispersion", "illuminant_residual"),
    "calibrate": ("clipping", "specular", "roi_area", "roi_dispersion"),
}


def new_request_id(now: datetime | None = None) -> str:
    """Generate a 26-character ULID-shaped request id.

    Args:
        now: Timestamp to encode (defaults to the current UTC time).

    Returns:
        Crockford base32 string: 10 characters of millisecond timestamp followed by
        16 characters of randomness, lexicographically sortable by creation time.
    """
    moment = now or datetime.now(timezone.utc)
    milliseconds = int(moment.timestamp() * 1000)
    time_part = ""
    for shift in range(45, -1, -5):
        time_part += _CROCKFORD[(milliseconds >> shift) & 0x1F]
    random_part = "".join(_CROCKFORD[byte & 0x1F] for byte in os.urandom(16))
    return (time_part + random_part)[: config.RESULT_ID_PREFIX_LENGTH]


def decode_image(image_bytes: bytes) -> NDArray[np.uint8]:
    """Decode an uploaded image to OpenCV BGR, honouring the EXIF orientation.

    Args:
        image_bytes: Raw uploaded bytes (JPEG/PNG/anything Pillow can open).

    Returns:
        ``(H, W, 3)`` ``uint8`` OpenCV image in **encoded** sRGB (BGR order).

    Raises:
        schemas.ApiError: ``BAD_IMAGE`` when the payload is not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except Exception as error:  # noqa: BLE001 - all decode failures are BAD_IMAGE
        raise schemas.ApiError(
            "BAD_IMAGE",
            f"无法解码上传的图片：{error}",
            "请上传手机或相机直接导出的 JPEG / PNG，不要上传 HEIC 或损坏的文件",
        ) from error
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise schemas.ApiError("BAD_IMAGE", "图片内容为空或通道数异常", "请重新拍摄并上传")
    return np.ascontiguousarray(array[:, :, ::-1])


def _gate_satisfaction(spec: GateSpec, value: float) -> float:
    """Map a measured value to a 0..1 satisfaction against its threshold.

    A gate scores 1.0 at or below ``GATE_FULL_MARK_RATIO * threshold`` for
    "at most" gates (at or above it for "at least" gates, capped at 1.0) and
    decays linearly to 0.0 at the threshold itself.

    Args:
        spec: The gate declaration.
        value: The measured value.

    Returns:
        Satisfaction in ``[0, 1]``.
    """
    if not spec.maximum:
        return float(np.clip(value / spec.threshold, 0.0, 1.0))
    full_mark = config.GATE_FULL_MARK_RATIO * spec.threshold
    span = spec.threshold - full_mark
    if span <= 0:
        return 1.0 if value <= spec.threshold else 0.0
    return float(np.clip((spec.threshold - value) / span, 0.0, 1.0))


def _gate_message(spec: GateSpec, value: float, passed: bool) -> str:
    """Render a gate message stating the measured value and the threshold."""
    if spec.id == "roi_area":
        rendered = f"{int(round(value))}"
        bound = ">=" if passed else "<"
        return f"{spec.label} {rendered}，{'满足' if passed else '低于'}门禁 {int(spec.threshold)}（实测{bound}阈值）"
    rendered = f"{value:.4f}" if spec.threshold < 1.0 else f"{value:.2f}"
    comparison = "低于" if passed else "超过"
    return f"{spec.label} {rendered}，{comparison}门禁 {spec.threshold:g}"


def build_confidence(
    mode: str,
    measured: dict[str, float],
    extra_caps: list[str],
    applied_profile: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Fuse gate measurements into ``confidence`` and collect warnings.

    Args:
        mode: ``nocard`` / ``card`` / ``calibrate``.
        measured: ``{gate_id: measured_value}``; missing gates are skipped.
        extra_caps: Confidence levels that must not be exceeded, from failures
            outside the numeric gates (e.g. a rejected camera profile).
        applied_profile: Whether a self-calibrated card profile was applied.

    Returns:
        ``(confidence, warnings)`` where ``confidence`` matches the contract's
        object and ``warnings`` explains any level reduction.
    """
    warnings: list[str] = []
    gates: list[schemas.GateResult] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for gate_id in _GATES_BY_MODE[mode]:
        if gate_id not in measured:
            continue
        spec = next(item for item in GATE_SPECS if item.id == gate_id)
        value = float(measured[gate_id])
        passed = value <= spec.threshold if spec.maximum else value >= spec.threshold
        satisfaction = _gate_satisfaction(spec, value)
        weight = config.GATE_WEIGHTS.get(gate_id, 1.0)
        weighted_sum += weight * satisfaction
        weight_total += weight
        gates.append(
            schemas.GateResult(
                id=spec.id,
                passed=bool(passed),
                value=float(value),
                threshold=float(spec.threshold),
                message=_gate_message(spec, value, bool(passed)),
            )
        )

    score = weighted_sum / weight_total if weight_total else 0.0
    if score >= config.CONFIDENCE_HIGH_MIN:
        level = "high"
    elif score >= config.CONFIDENCE_MEDIUM_MIN:
        level = "medium"
    elif score >= config.CONFIDENCE_LOW_MIN:
        level = "low"
    else:
        level = "insufficient"

    def cap(current: str, maximum: str, reason: str) -> str:
        if config.CONFIDENCE_ORDER.index(current) > config.CONFIDENCE_ORDER.index(maximum):
            warnings.append(reason)
            return maximum
        return current

    by_id = {gate.id: gate for gate in gates}
    if "roi_area" in by_id and not by_id["roi_area"].passed:
        level = cap(
            level,
            "insufficient",
            "有效皮肤像素不足，无法给出可信结论：请放大取景，让下颌或颈部皮肤占据更多画面，"
            "避开阴影与高光后重拍",
        )
    if mode == "card" and "ccm_delta_e" in by_id and not by_id["ccm_delta_e"].passed:
        level = cap(
            level,
            "low",
            "色卡标定残差过大：卡片可能反光、被遮挡或打印偏色，请重新拍摄或重新标定色卡",
        )
    if mode == "nocard" and "illuminant_residual" in by_id and not by_id["illuminant_residual"].passed:
        level = cap(
            level,
            "low",
            "光源估计残差过大（ILLUMINANT_UNRELIABLE）：环境光过杂或眼白/背景不可用，"
            "建议改用色卡模式",
        )
    mode_cap = config.MODE_MAX_CONFIDENCE.get(mode)
    if mode_cap is not None:
        level = cap(
            level,
            mode_cap,
            "无卡模式没有可自证的标定残差，置信度上限为 medium；"
            "需要更确定的结果请打印参考卡改用 card 模式",
        )
    if applied_profile and level == "high":
        warnings.append("本次结果使用了自标定色卡档案；档案只恢复相对颜色，不含绝对亮度")

    if level == "insufficient":
        warnings.append(
            "当前条件下测不准，因此不给出配色建议。请改在均匀自然光下、无直射反光时重拍，"
            "或打印参考卡使用 card 模式。"
        )

    return (
        {
            "level": level,
            "score": round(float(score), 4),
            "gates": gates,
        },
        warnings,
    )


def _resolve_skin_polygons(
    image_bgr: NDArray[np.uint8],
    rois: schemas.RoisSpec | None,
    faces: list[tuple[int, int, int, int]],
) -> tuple[list[list[list[float]]], list[str]]:
    """Return skin polygons and their labels, falling back to a Haar auto ROI.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image.
        rois: Client-supplied ROIs, or ``None``.
        faces: Already-detected face boxes (may be empty).

    Returns:
        ``(polygons, labels)`` in original image pixel coordinates.

    Raises:
        schemas.ApiError: ``FACE_NOT_FOUND`` when no ROI was supplied and Haar
            could not find a face.
    """
    if rois is not None and rois.skin:
        polygons = [list(polygon.points) for polygon in rois.skin]
        labels = [polygon.label for polygon in rois.skin]
        return polygons, labels
    if not faces:
        raise schemas.ApiError(
            "FACE_NOT_FOUND",
            "未提供 ROI，且 Haar 级联未在人脸附近检测到人脸",
            "请在前端手动框选下颌或颈部皮肤区域后重试",
        )
    auto = face_vision.auto_skin_polygons(image_bgr, faces[0])
    if not auto:
        raise schemas.ApiError(
            "ROI_TOO_SMALL",
            "自动推断的 ROI 落在画面之外",
            "请让整张脸完整入镜，或手动框选皮肤区域",
        )
    polygons = [list(item["points"]) for item in auto]  # type: ignore[arg-type]
    labels = [str(item["label"]) for item in auto]
    return polygons, labels


def _reference_mask(
    image_bgr: NDArray[np.uint8],
    rois: schemas.RoisSpec | None,
    faces: list[tuple[int, int, int, int]],
) -> NDArray[np.uint8]:
    """Build the exclude mask used for semantic illuminant candidate selection."""
    exclude = face_vision.face_mask(image_bgr, faces)
    if rois is not None:
        for group in (rois.skin, rois.reference):
            for polygon in group:
                exclude = np.maximum(
                    exclude,
                    roi_vision.polygons_to_mask(image_bgr.shape, [list(polygon.points)]),
                )
    return exclude


def _estimate_illuminant_nocard(
    image_bgr: NDArray[np.uint8],
    faces: list[tuple[int, int, int, int]],
    exclude_mask: NDArray[np.uint8],
    illuminant_guess: str,
) -> dict[str, Any]:
    """Estimate the scene illuminant from semantically masked achromatic surfaces.

    Sources, in decreasing reliability: sclera, teeth, near-neutral background.
    Skin, card and the whole face box are masked out first -- a global gray-world
    average is never computed, because on a face close-up it would neutralise the
    skin signal the service exists to measure.

    Args:
        image_bgr: ``(H, W, 3)`` OpenCV image in encoded sRGB.
        faces: Face boxes in original image pixels.
        exclude_mask: ``uint8`` mask (255 = exclude) of skin/card/face.
        illuminant_guess: The client's ``capture.illuminantGuess`` prior.

    Returns:
        The :func:`skintone.color.illuminant.solve_illuminant` dict, with the
        ``sources`` key added listing which candidate sources contributed.
    """
    sources: list[tuple[str, NDArray[np.float64], float]] = [
        ("sclera", face_vision.sclera_candidates(image_bgr, faces), 1.0),
        ("teeth", face_vision.teeth_candidates(image_bgr, faces), 0.9),
        ("background", roi_vision.neutral_candidates(image_bgr, exclude_mask), 0.6),
    ]
    parts: list[tuple[str, NDArray[np.float64], float]] = []
    for name, pixels, weight in sources:
        if pixels.shape[0] < config.ILLUMINANT_MIN_CANDIDATE_PIXELS:
            continue
        median_linear = np.median(pixels, axis=0)
        if float(np.sum(median_linear)) <= 1e-6:
            continue
        parts.append((name, np.asarray(xyz_to_xy(linear_rgb_to_xyz(median_linear))), weight))

    if not parts:
        # 采不到任何中性参考面：交给 solve_illuminant 用用户选的光源做回退，
        # 它会在 D65 与先验之间按 ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK 混合。
        solved = solve_illuminant(None, "d65-assumed", illuminant_guess, assumed_d65=True)
        solved["sources"] = []
        return solved

    weights = np.array([weight for _, _, weight in parts], dtype=np.float64)
    weights = weights / weights.sum()
    xy = np.sum(np.stack([xy for _, xy, _ in parts]) * weights[:, None], axis=0)
    solved = solve_illuminant(xy, "nocard-semantic", illuminant_guess)
    solved["sources"] = [name for name, _, _ in parts]
    return solved


def _skin_from_card(
    stats: RoiStats, matrix: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a CCM to the ROI median colour.

    Args:
        stats: ROI statistics (linear-sRGB median).
        matrix: 3x3 or 3x4 CCM mapping linear sRGB to CIE XYZ (D65).

    Returns:
        ``(d65_linear_rgb, d65_lab)`` -- the ROI colour in the D65 frame.
    """
    assert stats.median_linear_rgb is not None
    xyz = card_vision.apply_ccm(stats.median_linear_rgb, matrix)
    return np.asarray(xyz_to_linear_rgb(xyz)), np.asarray(xyz_to_lab(xyz))


def _skin_from_nocard(
    stats: RoiStats, illuminant_xy: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bradford-adapt the ROI colour from the estimated illuminant to D65.

    Args:
        stats: ROI statistics (linear-sRGB median, camera-referenced).
        illuminant_xy: Corrected illuminant chromaticity ``[x, y]``.

    Returns:
        ``(d65_linear_rgb, d65_lab)``.
    """
    assert stats.median_linear_rgb is not None
    source_white = white_xyz_from_xy(illuminant_xy)
    xyz = linear_rgb_to_xyz(stats.median_linear_rgb)
    adapted = adapt_xyz(xyz, source_white, D65_XYZ, DEFAULT_ADAPTATION)
    return np.asarray(xyz_to_linear_rgb(adapted)), np.asarray(xyz_to_lab(adapted))


def _skin_block(
    stats: RoiStats,
    d65_linear_rgb: NDArray[np.float64],
    d65_lab: NDArray[np.float64],
) -> dict[str, Any]:
    """Assemble the contract's ``skin`` object from a measured ROI colour."""
    lab = [round(float(value), 2) for value in d65_lab]
    linear = [round(float(value), 4) for value in np.clip(d65_linear_rgb, 0.0, 1.0)]
    classification = undertone(float(d65_lab[0]), float(d65_lab[1]), float(d65_lab[2]))
    melanin, hemoglobin = melanin_hemoglobin(
        np.clip(d65_linear_rgb, config.OD_LINEAR_FLOOR, 1.0)
    )
    axis = classification["axis"]
    assert isinstance(axis, dict)
    probabilities = classification["probabilities"]
    assert isinstance(probabilities, dict)
    hex_value = linear_to_hex(np.clip(d65_linear_rgb, 0.0, 1.0))
    return {
        "roi": {"pixelCount": int(stats.valid_pixels), "regions": list(stats.regions)},
        "labD65": {"L": lab[0], "a": lab[1], "b": lab[2]},
        "linearRgb": linear,
        "hex": hex_value,
        "itaDeg": round(ita_degrees(float(d65_lab[0]), float(d65_lab[2])), 2),
        "depthClass": depth_class(float(d65_lab[0]), float(d65_lab[2])),
        "hueAngleDeg": round(hue_angle_deg(float(d65_lab[1]), float(d65_lab[2])), 2),
        "chroma": round(float(np.hypot(d65_lab[1], d65_lab[2])), 2),
        "undertone": {
            "label": classification["label"],
            "axis": {
                "aOverB": None
                if axis.get("aOverB") is None
                else round(float(axis["aOverB"]), 4),
                "hueAngleDeg": round(float(axis["hueAngleDeg"]), 2),
            },
            "probabilities": {
                key: round(float(value), 4) for key, value in probabilities.items()
            },
        },
        "melaninIndex": round(melanin, 4),
        "hemoglobinIndex": round(hemoglobin, 4),
    }


def _empty_skin_block(stats: RoiStats) -> dict[str, Any]:
    """Return a ``skin`` object with every unmeasurable field set to ``null``."""
    return {
        "roi": {"pixelCount": int(stats.valid_pixels), "regions": list(stats.regions)},
        "labD65": None,
        "linearRgb": None,
        "hex": None,
        "itaDeg": None,
        "depthClass": None,
        "hueAngleDeg": None,
        "chroma": None,
        "undertone": None,
        "melaninIndex": None,
        "hemoglobinIndex": None,
    }


def _profile_reference(
    samples: list[card_vision.PatchSample], profile: dict[str, list[int]]
) -> list[card_vision.PatchSample]:
    """Replace each sample's nominal value with a card profile's measured value.

    Args:
        samples: Patch samples from the photo being analysed.
        profile: ``{patch_id: [r, g, b]}`` measured under the user's own daylight.

    Returns:
        New samples whose ``reference_xyz``/``reference_lab`` come from the profile.
    """
    replaced: list[card_vision.PatchSample] = []
    for sample in samples:
        measured = profile.get(sample.id)
        if measured is None:
            replaced.append(sample)
            continue
        reference_srgb = np.asarray(measured, dtype=np.float64)
        replaced.append(
            card_vision.PatchSample(
                id=sample.id,
                kind=sample.kind,
                measured_srgb8=sample.measured_srgb8,
                measured_linear=sample.measured_linear,
                nominal_srgb=reference_srgb,
                reference_xyz=linear_rgb_to_xyz(srgb8_to_linear(reference_srgb)),
                reference_lab=np.asarray(
                    xyz_to_lab(linear_rgb_to_xyz(srgb8_to_linear(reference_srgb)))
                ),
                usable=sample.usable,
            )
        )
    return replaced


def calibrate_card(
    image_bgr: NDArray[np.uint8],
    meta: schemas.AnalyzeMeta,
    profile_id: str,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Run the self-calibration flow of contract section 2.4.

    Args:
        image_bgr: Decoded photo of the printed card in the user's own daylight.
        meta: Request metadata (card id, illuminant guess).
        profile_id: Client-chosen profile id.
        now: Creation timestamp.

    Returns:
        ``(response_dict, profile)`` where ``profile`` is
        ``{patch_id: [r, g, b]}`` in **encoded** sRGB and is what gets persisted
        for later ``cardProfileId`` use.

    Raises:
        schemas.ApiError: ``CARD_NOT_DETECTED`` when the four markers are missing
            or too few patches could be sampled.
    """
    card_id = meta.cardId or config.CARD_ID
    if card_id not in definitions.CARD_IDS:
        raise schemas.ApiError(
            "CARD_NOT_DETECTED",
            f"未知的色卡 id：{card_id}",
            f"当前服务支持的色卡：{', '.join(definitions.CARD_IDS)}",
        )
    markers = card_vision.detect_markers(image_bgr)
    homography = card_vision.homography_to_canonical(markers)
    if homography is None:
        raise schemas.ApiError(
            "CARD_NOT_DETECTED",
            f"只找到 {len(markers)} 个 ArUco 标记，需要全部 4 个（id 0-3）",
            "把参考卡平放在脸颊旁，让四个正方形黑框完整入镜且不反光",
        )
    warped = card_vision.warp_to_canonical(image_bgr, homography)
    samples = card_vision.sample_all_patches(warped)
    usable = [sample for sample in samples if sample.usable]
    if len(usable) < config.CCM_MIN_PATCHES:
        raise schemas.ApiError(
            "CARD_NOT_DETECTED",
            f"只能稳定采到 {len(usable)} 个色块，需要至少 {config.CCM_MIN_PATCHES} 个",
            "让卡片完整入镜、避免高光过曝，并在均匀光照下重拍",
        )

    neutrals = [
        sample
        for sample in usable
        if sample.kind == "gray" and float(np.mean(sample.measured_linear)) > 0.05
    ]
    if not neutrals:
        raise schemas.ApiError(
            "CARD_NOT_DETECTED",
            "灰阶条全部过曝或过暗，无法锁定中性轴",
            "降低曝光或换到光线柔和的位置重拍色卡",
        )

    measured_stack = np.stack([sample.measured_linear for sample in neutrals])
    chroma_ratios = measured_stack / np.clip(
        measured_stack.sum(axis=1, keepdims=True), 1e-9, None
    )
    paper_tint = np.median(chroma_ratios, axis=0)
    white_balance = (1.0 / 3.0) / np.clip(paper_tint, 1e-6, None)

    brightest = max(
        neutrals, key=lambda sample: float(np.mean(sample.reference_xyz))
    )
    target = srgb8_to_linear(np.asarray(brightest.nominal_srgb, dtype=np.float64))
    normalized_brightest = brightest.measured_linear * white_balance
    scale = float(np.mean(target) / max(np.mean(normalized_brightest), 1e-6))

    gain = white_balance * scale
    corrected = {
        sample.id: np.clip(sample.measured_linear * gain, 0.0, 1.0) for sample in usable
    }
    measured_srgb = {
        patch_id: [int(round(float(value))) for value in linear_to_srgb8(value)]
        for patch_id, value in corrected.items()
    }

    neutral_chroma = [
        float(np.hypot(*xyz_to_lab(linear_rgb_to_xyz(corrected[sample.id]))[1:]))
        for sample in neutrals
    ]
    max_abs_chroma = round(max(neutral_chroma), 3) if neutral_chroma else 0.0
    neutrality_passed = max_abs_chroma <= config.CALIBRATION_GRAY_MAX_CHROMA

    best_mean = None
    for extra in np.linspace(
        config.CALIBRATION_LUMINANCE_SEARCH[0],
        config.CALIBRATION_LUMINANCE_SEARCH[1],
        config.CALIBRATION_LUMINANCE_SEARCH[2],
    ):
        labs = np.stack(
            [
                xyz_to_lab(linear_rgb_to_xyz(np.clip(corrected[sample.id] * extra, 0.0, 1.0)))
                for sample in usable
            ]
        )
        references = np.stack([sample.reference_lab for sample in usable])
        mean = float(np.mean(ciede2000(references, labs)))
        if best_mean is None or mean < best_mean:
            best_mean = mean

    quality_level = "good"
    if best_mean is None or best_mean > config.CALIBRATION_FAIR_DELTA_E:
        quality_level = "poor"
    elif best_mean > config.CALIBRATION_GOOD_DELTA_E:
        quality_level = "fair"

    return (
        {
            "profileId": profile_id,
            "cardId": config.CARD_ID,
            "createdAt": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "measuredSrgb": measured_srgb,
            "grayNeutrality": {
                "maxAbsChroma": max_abs_chroma,
                "passed": bool(neutrality_passed),
            },
            "quality": {
                "level": quality_level,
                "deltaE00MeanAfter": None
                if best_mean is None
                else round(float(best_mean), 3),
            },
            "note": "标定只恢复相对颜色（整体光照增益不可知）；中性轴由灰阶条锁定。",
        },
        measured_srgb,
    )


def analyze(
    image_bytes: bytes,
    meta: schemas.AnalyzeMeta,
    rois: schemas.RoisSpec | None,
    profile_lookup: Any = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Run the full analysis pipeline for one request.

    Args:
        image_bytes: Raw uploaded image bytes.
        meta: Parsed ``meta`` multipart part.
        rois: Parsed ``rois`` part, or ``None`` for a Haar auto ROI.
        profile_lookup: Callable ``(profile_id) -> dict | None`` returning a stored
            card profile, used when ``meta.cardProfileId`` is set.
        now: Processing timestamp (defaults to now, UTC).

    Returns:
        ``(payload, image_to_store)``. ``image_to_store`` is the original bytes
        when the user consented to storage, otherwise ``None`` -- in which case
        nothing is ever written to disk.

    Raises:
        schemas.ApiError: for any of the contract's error codes.
    """
    moment = now or datetime.now(timezone.utc)
    image_bgr = decode_image(image_bytes)
    needs_faces = (rois is None or not rois.skin) or meta.mode == "nocard"
    faces = face_vision.detect_faces(image_bgr) if needs_faces else []
    polygons, labels = _resolve_skin_polygons(image_bgr, rois, faces)
    stats = roi_vision.compute_roi_stats(image_bgr, polygons, labels)

    warnings: list[str] = []
    measured_gates: dict[str, float] = {
        "clipping": stats.clipping_ratio,
        "specular": stats.specular_ratio,
        "roi_area": float(stats.valid_pixels),
        "roi_dispersion": float(stats.dispersion or 0.0),
    }

    calibration_block: dict[str, Any] | None = None
    applied_profile = False

    if meta.mode == "card":
        card_id = meta.cardId or config.CARD_ID
        if card_id not in definitions.CARD_IDS:
            raise schemas.ApiError(
                "CARD_NOT_DETECTED",
                f"未知的色卡 id：{card_id}",
                f"当前服务支持的色卡：{', '.join(definitions.CARD_IDS)}",
            )
        markers = card_vision.detect_markers(image_bgr)
        homography = card_vision.homography_to_canonical(markers)
        if homography is None:
            raise schemas.ApiError(
                "CARD_NOT_DETECTED",
                f"在人脸附近只找到 {len(markers)} 个 ArUco 标记，需要全部 4 个（id 0-3）",
                "把参考卡平放在脸颊旁，让正方形黑框完整入镜且不反光",
            )
        warped = card_vision.warp_to_canonical(image_bgr, homography)
        samples = card_vision.sample_all_patches(warped)

        reference_samples = samples
        if meta.cardProfileId:
            profile = profile_lookup(meta.cardProfileId) if profile_lookup else None
            if profile is None:
                raise schemas.ApiError(
                    "CARD_PROFILE_NOT_FOUND",
                    f"未找到色卡自标定档案 {meta.cardProfileId}",
                    "请先在 /v1/card/skintone-a4-v1/calibrate 用你自己的日光标定这张纸",
                )
            reference_samples = _profile_reference(samples, profile)
            applied_profile = True
            warnings.append(
                f"已套用色卡自标定档案 {meta.cardProfileId}（只恢复相对颜色）"
            )

        try:
            ccm = card_vision.choose_ccm(reference_samples)
        except ValueError as error:
            raise schemas.ApiError(
                "CARD_NOT_DETECTED",
                f"可用于解 CCM 的色块不足：{error}",
                "避免过曝与反光，让整张卡片清晰入镜后在均匀光照下重拍",
            ) from error

        measured_gates["ccm_delta_e"] = ccm.delta_e_mean
        calibration_block = {
            "ccm": [[round(float(value), 6) for value in row] for row in ccm.matrix],
            "ccmKind": ccm.kind,
            "deltaE00Mean": round(ccm.delta_e_mean, 4),
            "deltaE00Max": round(ccm.delta_e_max, 4),
            "perPatch": [
                {"id": patch_id, "deltaE00": round(value, 4)}
                for patch_id, value in ccm.per_patch.items()
            ],
        }

        white_xy = card_vision.card_white_xy(samples)
        # 比色卡模式刻意禁用先验：卡已经真的测出了光源，用户的猜测掺进去只会更差。
        illuminant = solve_illuminant(
            white_xy, "card", meta.capture.illuminantGuess, prior_weight=0.0
        )

        if stats.median_linear_rgb is None:
            skin_block = _empty_skin_block(stats)
            warnings.append("ROI 内没有可用像素，未给出肤色数值")
        else:
            d65_linear, d65_lab = _skin_from_card(stats, ccm.matrix)
            skin_block = _skin_block(stats, d65_linear, d65_lab)

    elif meta.mode == "nocard":
        exclude = _reference_mask(image_bgr, rois, faces)
        illuminant = _estimate_illuminant_nocard(
            image_bgr, faces, exclude, meta.capture.illuminantGuess
        )
        sources = illuminant.pop("sources", [])
        measured_gates["illuminant_residual"] = abs(float(illuminant["duv"] or 0.0))
        if not sources:
            # 采不到中性面时不再一刀切按 D65：若用户选了具体光源，会以更高权重
            # 把先验与 D65 混合（见 color/illuminant.py），所以提示语也要跟上。
            warnings.append(
                "未找到眼白/牙齿/背景等非皮肤中性面，已按你选择的拍摄光源估算色温"
                if illuminant.get("priorApplied")
                else "未找到眼白/牙齿/背景等非皮肤中性面，光源色温按 D65 处理"
            )
        else:
            warnings.append("光源估计使用的候选来源：" + "、".join(sources))
        warnings.append(
            "无卡模式依赖眼白、牙齿与背景的近似中性假设，这些表面并非理想白体，"
            "系统会把估计约束到普朗克/日光轨迹上以降低自由度"
        )
        if stats.median_linear_rgb is None:
            skin_block = _empty_skin_block(stats)
            warnings.append("ROI 内没有可用像素，未给出肤色数值")
        else:
            d65_linear, d65_lab = _skin_from_nocard(stats, list(illuminant["xy"]))  # type: ignore[arg-type]
            skin_block = _skin_block(stats, d65_linear, d65_lab)
    else:  # pragma: no cover - calibrate is handled by a dedicated endpoint
        raise schemas.ApiError("BAD_IMAGE", "calibrate 模式请使用 /calibrate 端点", "")

    confidence, gate_warnings = build_confidence(
        meta.mode, measured_gates, [], applied_profile
    )
    warnings.extend(gate_warnings)

    advice: dict[str, Any] | None = None
    if confidence["level"] != "insufficient" and skin_block["labD65"] is not None:
        lab = np.array(
            [
                skin_block["labD65"]["L"],
                skin_block["labD65"]["a"],
                skin_block["labD65"]["b"],
            ],
            dtype=np.float64,
        )
        advice, advice_warnings = palette_rules.build_advice(lab, meta.hairLStar)
        warnings.extend(advice_warnings)
    elif confidence["level"] == "insufficient":
        advice = None

    if meta.capture.flash:
        warnings.append("检测到使用闪光灯：闪光会改变光源色温，建议关闭后重拍")
    if not meta.capture.wbLocked:
        warnings.append("相机白平衡未锁定：自动白平衡会削弱色偏信号，请尽量锁定白平衡")

    payload: dict[str, Any] = {
        "requestId": "",
        "specVersion": config.SPEC_VERSION,
        "serverVersion": config.SERVER_VERSION,
        "createdAt": moment.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "mode": meta.mode,
        "confidence": confidence,
        "illuminant": {
            "method": illuminant.get("method", "unknown"),
            "cct": None if illuminant.get("cct") is None else round(float(illuminant["cct"]), 1),
            "duv": None if illuminant.get("duv") is None else round(float(illuminant["duv"]), 5),
            "xy": None
            if illuminant.get("xy") is None
            else [round(float(value), 5) for value in illuminant["xy"]],
            "adaptation": DEFAULT_ADAPTATION,
            "assumedD65": bool(illuminant.get("assumedD65", False)),
            # 让"用户选的光源到底有没有生效"可查。
            # 注意：schema 里这三个字段带默认值，所以漏了透传也不会报错——
            # 只有断言 priorApplied 真正为 True 的测试才拦得住（见 test_api.py）。
            "locus": illuminant.get("locus"),
            "priorApplied": bool(illuminant.get("priorApplied", False)),
            "priorWeight": None
            if illuminant.get("priorWeight") is None
            else round(float(illuminant["priorWeight"]), 3),
        },
        "calibration": calibration_block,
        "skin": skin_block,
        "advice": advice,
        "warnings": warnings,
        "disclaimer": config.DISCLAIMER,
    }
    image_to_store = image_bytes if meta.consent.storeImage else None
    return payload, image_to_store


def uptime_seconds(started_at: float) -> float:
    """Return seconds elapsed since ``started_at`` (a ``time.monotonic()`` value)."""
    return round(time.monotonic() - started_at, 3)

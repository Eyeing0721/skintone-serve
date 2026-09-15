"""All request/response models for the HTTP API (contract section 2).

JSON field names are camelCase exactly as the contract spells them. Response
models never omit keys: a value that cannot be measured is ``null``
(contract section 2.3, "字段全部必现").

Colour-space conventions for every field below (contract section 4):

* ``labD65`` / ``lab`` -- CIELAB, D65 white point, 2-degree observer.
* ``linearRgb``        -- linear (un-encoded) sRGB primaries, 0..1.
* ``nominalSrgb``      -- encoded sRGB, 0..255.
* ``xy``               -- CIE 1931 chromaticity.
* ``cct``              -- kelvin. ``duv`` -- signed offset in CIE 1960 uv.
* ``itaDeg``           -- Individual Typology Angle in degrees.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnalyzeMode = Literal["nocard", "card", "calibrate"]
PatchKind = Literal["gray", "color", "skin", "olive"]
ConfidenceLevel = Literal["high", "medium", "low", "insufficient"]
DepthClass = Literal[
    "very-light", "light", "intermediate", "tan", "brown", "dark"
]
UndertoneLabel = Literal[
    "cool", "neutral-cool", "neutral", "neutral-warm", "warm", "olive"
]
IlluminantGuess = Literal[
    "screen", "daylight", "shade", "tungsten", "fluorescent", "led", "unknown"
]


class _Model(BaseModel):
    """Base model.

    Unknown keys are ignored on input (a newer client may send extra fields) but
    never emitted on output, because response models are constructed field by
    field by this server.
    """

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorBody(_Model):
    """Machine-readable error payload (contract section 2.3)."""

    code: str
    message: str
    hint: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(_Model):
    """Envelope for every non-2xx JSON response."""

    error: ErrorBody


# ---------------------------------------------------------------------------
# Health / card spec / stats
# ---------------------------------------------------------------------------


class HealthResponse(_Model):
    """``GET /v1/health`` payload."""

    status: str
    version: str
    specVersion: str
    uptimeSeconds: float
    cards: list[str]
    authRequired: bool


class PaperSpec(_Model):
    """Physical paper description of a reference card."""

    name: str
    widthMm: float
    heightMm: float
    dpi: int


class MarkersSpec(_Model):
    """ArUco marker layout of a reference card."""

    dictionary: str
    sizeMm: float
    ids: list[int]
    centersMm: list[list[float]]


class PatchSpec(_Model):
    """One printable patch of a reference card."""

    id: str
    kind: PatchKind
    centerMm: list[float]
    sizeMm: float
    nominalSrgb: list[int]


class CardSpec(_Model):
    """``GET /v1/card/{card_id}`` payload -- the card geometry, code-ified."""

    cardId: str
    specVersion: str
    paper: PaperSpec
    markers: MarkersSpec
    patches: list[PatchSpec]
    instructions: list[str]


class StatsResponse(_Model):
    """``GET /v1/stats`` payload; aggregated counters only, never user data."""

    totalRequests: int
    storedImages: int
    diskBytes: int


# ---------------------------------------------------------------------------
# Request metadata
# ---------------------------------------------------------------------------


class CaptureMeta(_Model):
    """Capture-side hints supplied by the client."""

    wbLocked: bool = False
    raw: bool = False
    flash: bool = False
    illuminantGuess: IlluminantGuess = "unknown"
    devicePixelRatio: float | None = None


class ConsentMeta(_Model):
    """Explicit user consent flags (contract section 7)."""

    storeImage: bool = True
    acceptedAt: str | None = None


class AnalyzeMeta(_Model):
    """The ``meta`` multipart part of ``POST /v1/analyze``."""

    mode: AnalyzeMode
    cardId: str | None = None
    cardProfileId: str | None = None
    hairLStar: float | None = None
    capture: CaptureMeta = Field(default_factory=CaptureMeta)
    consent: ConsentMeta = Field(default_factory=ConsentMeta)
    clientVersion: str | None = None


class RoiPolygon(_Model):
    """A closed polygon in **original image pixel** coordinates."""

    label: str
    points: list[list[float]]


class RoisSpec(_Model):
    """The ``rois`` multipart part; coordinates are original-image pixels."""

    skin: list[RoiPolygon] = Field(default_factory=list)
    reference: list[RoiPolygon] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------


class GateResult(_Model):
    """One confidence gate with its measured value and threshold."""

    id: str
    passed: bool
    value: float
    threshold: float
    message: str


class Confidence(_Model):
    """Fused confidence block (contract section 5)."""

    level: ConfidenceLevel
    score: float
    gates: list[GateResult]


class IlluminantInfo(_Model):
    """Estimated scene illuminant after locus projection."""

    method: str
    cct: float | None
    duv: float | None
    xy: list[float] | None
    adaptation: str
    assumedD65: bool
    #: 投影到哪条物理轨迹上：``planckian`` / ``daylight`` / ``None``。
    locus: str | None = None
    #: 用户选的光源先验是否真的改变了结果（选了"不确定"时为 False）。
    priorApplied: bool = False
    #: 先验拿到的权重；0 表示被忽略。
    priorWeight: float | None = None


class PerPatchDelta(_Model):
    """Per-patch calibration residual."""

    id: str
    deltaE00: float


class CalibrationInfo(_Model):
    """Colour-correction matrix and its residual, card modes only."""

    ccm: list[list[float]]
    ccmKind: str
    deltaE00Mean: float
    deltaE00Max: float
    perPatch: list[PerPatchDelta]


class SkinRoiInfo(_Model):
    """What the ROI stage actually measured."""

    pixelCount: int
    regions: list[str]


class LabValue(_Model):
    """CIELAB triple, D65 white point."""

    L: float
    a: float
    b: float


class UndertoneAxis(_Model):
    """Raw, interpretable axes behind the undertone label."""

    aOverB: float | None
    hueAngleDeg: float | None


class Undertone(_Model):
    """Warm/cool/olive classification with heuristic class weights."""

    label: UndertoneLabel
    axis: UndertoneAxis
    probabilities: dict[str, float]


class SkinInfo(_Model):
    """The measured skin block of the response."""

    roi: SkinRoiInfo
    labD65: LabValue | None
    linearRgb: list[float] | None
    hex: str | None
    itaDeg: float | None
    depthClass: DepthClass | None
    hueAngleDeg: float | None
    chroma: float | None
    undertone: Undertone | None
    melaninIndex: float | None
    hemoglobinIndex: float | None


class PaletteItem(_Model):
    """One recommended colour."""

    hex: str
    lab: list[float]
    name: str


class AvoidItem(_Model):
    """One colour to avoid, with the numeric reason."""

    hex: str
    reason: str


class Advice(_Model):
    """Garment-colour advice derived from the measured skin (contract 6)."""

    contrastLevel: str
    palette: list[PaletteItem]
    avoid: list[AvoidItem]


class AnalyzeResponse(_Model):
    """``POST /v1/analyze`` payload."""

    requestId: str
    specVersion: str
    serverVersion: str
    createdAt: str
    mode: AnalyzeMode
    confidence: Confidence
    illuminant: IlluminantInfo
    calibration: CalibrationInfo | None
    skin: SkinInfo
    advice: Advice | None
    warnings: list[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# Calibration endpoint
# ---------------------------------------------------------------------------


class GrayNeutrality(_Model):
    """How well the measured gray ramp held its neutrality."""

    maxAbsChroma: float
    passed: bool


class CalibrateQuality(_Model):
    """Quality of the self-calibration fit."""

    level: str
    deltaE00MeanAfter: float | None


class CalibrateResponse(_Model):
    """``POST /v1/card/{card_id}/calibrate`` payload."""

    profileId: str
    cardId: str
    createdAt: str
    measuredSrgb: dict[str, list[int]]
    grayNeutrality: GrayNeutrality
    quality: CalibrateQuality
    note: str


class DeleteResponse(_Model):
    """Body of a successful deletion (the endpoint returns 204 with no body)."""

    deleted: bool


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

#: Every error code the contract allows, with its HTTP status.
ERROR_STATUS: dict[str, int] = {
    "BAD_IMAGE": 400,
    "CARD_NOT_DETECTED": 400,
    "CARD_PROFILE_NOT_FOUND": 404,
    "ROI_TOO_SMALL": 400,
    "FACE_NOT_FOUND": 400,
    "ILLUMINANT_UNRELIABLE": 422,
    "PAYLOAD_TOO_LARGE": 413,
    "UNAUTHORIZED": 401,
    "RATE_LIMITED": 429,
    "INTERNAL": 500,
}


class ApiError(Exception):
    """An error that maps onto the contract's ``{"error": {...}}`` envelope.

    Attributes:
        code: One of :data:`ERROR_STATUS`.
        message: Human-readable explanation of what went wrong.
        hint: Actionable advice for the user.
        details: Optional structured extras (never user data).
        status_code: HTTP status derived from ``code`` unless overridden.
    """

    def __init__(
        self,
        code: str,
        message: str,
        hint: str = "",
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code if code in ERROR_STATUS else "INTERNAL"
        self.message = message
        self.hint = hint
        self.details = details or {}
        self.status_code = status_code or ERROR_STATUS[self.code]

    def to_response(self) -> ErrorResponse:
        """Render this error as the contract's error envelope."""
        return ErrorResponse(
            error=ErrorBody(
                code=self.code,
                message=self.message,
                hint=self.hint,
                details=self.details,
            )
        )

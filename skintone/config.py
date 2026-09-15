"""Central configuration: environment settings **and** every numeric constant.

Contract 8 requires that all thresholds and constants live here instead of being
scattered through function bodies. Nothing in this module depends on the rest of
the package, so it can be imported from anywhere.

Naming convention
-----------------
* ``*_MM``       -- millimetres on the printed card (physical geometry).
* ``*_PX``       -- pixels in the canonical card rasterisation.
* ``*_THRESHOLD``-- a gate threshold as written in contract section 5.
* ``*_WEIGHT``   -- a weight used by a scoring/fusion formula.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Contract identity (section 2.1)
# ---------------------------------------------------------------------------

SPEC_VERSION = "1.0.0"
SERVER_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Card geometry -- contract section 3, "skintone-a4-v1"
# ---------------------------------------------------------------------------

CARD_ID = "skintone-a4-v1"

PAPER_NAME = "A4"
PAPER_WIDTH_MM = 210.0
PAPER_HEIGHT_MM = 297.0
CARD_DPI = 300

#: Canonical rasterisation of the card: 210mm x 297mm at 300 DPI.
CARD_WIDTH_PX = 2480
CARD_HEIGHT_PX = 3508
#: 2480 px / 210 mm -- the single mm<->px scaling factor for the canonical card.
CARD_PX_PER_MM = CARD_WIDTH_PX / PAPER_WIDTH_MM

ARUCO_DICTIONARY = "DICT_4X4_50"
ARUCO_MARKER_IDS = (0, 1, 2, 3)
ARUCO_MARKER_SIZE_MM = 20.0
ARUCO_MARKER_CENTERS_MM = ((25.0, 25.0), (185.0, 25.0), (25.0, 272.0), (185.0, 272.0))

GRAY_PATCH_SIZE_MM = 26.0
COLOR_PATCH_SIZE_MM = 22.0
GRAY_PATCH_CENTER_Y_MM = 65.0
GRAY_PATCH_CENTER_X_MM = (45.0, 77.0, 109.0, 141.0, 173.0)
COLOR_PATCH_COL_X_MM = (31.0, 61.0, 91.0, 121.0, 151.0, 181.0)
COLOR_PATCH_ROW_Y_MM = (101.0, 131.0, 161.0, 191.0)

# ---------------------------------------------------------------------------
# Patch sampling -- contract section 5A step 2: centre 60 % region, median
# ---------------------------------------------------------------------------

PATCH_SAMPLE_FRACTION = 0.60
#: Minimum patch side, in canonical card pixels, required to sample at all.
PATCH_MIN_SAMPLE_PX = 4
#: A patch whose median encoded sRGB saturates in any channel carries no colour
#: information and is dropped from the CCM fit.
CCM_DROP_SATURATED_SRGB8 = 250
#: Minimum number of usable patches required to solve a CCM.
CCM_MIN_PATCHES = 12
#: A camera with a lifted black level needs the 3x4 (offset) CCM. The black
#: reference patch (nominal #000000) is used to detect that: if its measured
#: *linear* value exceeds this, the offset model is selected.
CCM_OFFSET_BLACK_LEVEL = 0.010

# ---------------------------------------------------------------------------
# Colour-difference / calibration gates -- contract section 5
# ---------------------------------------------------------------------------

#: Gate ``ccm_delta_e``: mean per-patch CIEDE2000 residual of the fitted CCM.
CCM_DELTA_E_MAX = 5.0

#: Gate ``clipping``: fraction of ROI pixels with any channel >= 250 or <= 5.
CLIP_HIGH_SRGB8 = 250
CLIP_LOW_SRGB8 = 5
CLIP_RATIO_MAX = 0.02

#: Gate ``specular``: fraction of ROI pixels whose CIELAB L* exceeds this.
SPECULAR_L_MAX = 85.0
SPECULAR_RATIO_MAX = 0.08

#: Gate ``roi_area``: number of usable skin pixels after rejection.
ROI_MIN_PIXELS = 4000

#: Gate ``roi_dispersion``: robust (median absolute deviation based) spread of
#: ROI pixels in CIELAB, expressed as a pseudo CIEDE2000 scale.
ROI_DISPERSION_MAX = 8.0

#: Gate ``illuminant_residual`` (contract 5B): |Duv| of the estimated illuminant
#: after projection onto the Planckian / CIE daylight locus.
ILLUMINANT_DUV_MAX = 0.02

#: Robust statistics: fraction trimmed from each tail when computing the trimmed
#: mean of an ROI (0.10 -> central 80 % is kept).
ROI_TRIM_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Confidence fusion -- contract section 5, "confidence synthesis"
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH_MIN = 0.75
CONFIDENCE_MEDIUM_MIN = 0.50
CONFIDENCE_LOW_MIN = 0.25

#: Per-gate weights used by the weighted mean in ``confidence.score``.
GATE_WEIGHTS = {
    "ccm_delta_e": 1.0,
    "clipping": 1.0,
    "specular": 1.0,
    "roi_area": 1.0,
    "roi_dispersion": 1.0,
    "illuminant_residual": 1.0,
}

#: Satisfaction curve: a "max" gate scores 1.0 at ``value <= FULL_MARK`` and 0.0
#: at ``value >= threshold``, linearly in between. FULL_MARK = ratio * threshold.
GATE_FULL_MARK_RATIO = 0.40

#: Hard ceiling on ``confidence.level`` per mode. The no-card route has no
#: self-proving calibration residual, so it may never report "high"
#: (contract section 9.4).
MODE_MAX_CONFIDENCE: dict[str, str] = {
    "nocard": "medium",
    "card": "high",
    "calibrate": "high",
}

#: Confidence levels ordered from worst to best; used to apply ceilings.
CONFIDENCE_ORDER = ("insufficient", "low", "medium", "high")

# ---------------------------------------------------------------------------
# ITA depth classes -- contract section 2.3 (lower bound inclusive, upper exclusive)
# ---------------------------------------------------------------------------

#: (class name, inclusive lower bound in degrees). Ordered from light to dark.
ITA_CLASSES = (
    ("very-light", 55.0),
    ("light", 41.0),
    ("intermediate", 28.0),
    ("tan", 10.0),
    ("brown", -30.0),
    ("dark", float("-inf")),
)

# ---------------------------------------------------------------------------
# Undertone heuristics -- contract section 0 / section 5C
# ---------------------------------------------------------------------------

#: The warm/cool axis is driven by the CIELAB hue angle h_ab of the skin:
#: golden/yellow skin sits at a larger h_ab than pink/rosy skin.
#: axis = tanh((h_ab - HUE_NEUTRAL_DEG) / HUE_AXIS_SCALE_DEG), in [-1, 1].
UNDERTONE_HUE_NEUTRAL_DEG = 48.0
UNDERTONE_HUE_AXIS_SCALE_DEG = 12.0

#: Label cut points on the warm/cool axis (>= warm, >= neutral-warm, ...).
UNDERTONE_LABEL_CUTS = (
    ("warm", 0.70),
    ("neutral-warm", 0.30),
    ("neutral", -0.30),
    ("neutral-cool", -0.70),
)
UNDERTONE_COOL_LABEL = "cool"

#: Softmax temperature (in axis units squared) for the class probabilities.
UNDERTONE_SOFTMAX_TAU = 0.5
#: Axis prototypes for the cool / neutral / warm classes.
UNDERTONE_AXIS_PROTOTYPES = {"cool": -0.70, "neutral": 0.00, "warm": 0.70}
#: Olive evidence: high hue angle + low a*/b* + low chroma.
UNDERTONE_OLIVE_HUE_CENTER = 60.0
UNDERTONE_OLIVE_HUE_SCALE = 5.0
UNDERTONE_OLIVE_RATIO_CENTER = 0.62
UNDERTONE_OLIVE_RATIO_SCALE = 0.06
UNDERTONE_OLIVE_CHROMA_CENTER = 26.0
UNDERTONE_OLIVE_CHROMA_SCALE = 5.0
#: logit = OLIVE_LOGIT_GAIN * olive_evidence + OLIVE_LOGIT_BIAS
UNDERTONE_OLIVE_LOGIT_GAIN = 4.0
UNDERTONE_OLIVE_LOGIT_BIAS = -1.6
#: Olive label override.
UNDERTONE_OLIVE_LABEL_MIN_PROB = 0.35
UNDERTONE_OLIVE_LABEL_MIN_EVIDENCE = 0.50

# ---------------------------------------------------------------------------
# Log-chromaticity melanin / haemoglobin decomposition -- contract section 5C
# ---------------------------------------------------------------------------

#: Effective centre wavelengths (nm) of the three sRGB camera channels; used to
#: document where the two chromophore axes below come from.
OD_CHANNEL_WAVELENGTHS_NM = (610.0, 540.0, 460.0)
#: Melanin axis in linear-sRGB optical-density space. Melanin absorbs broadly and
#: monotonically (Jacques 2013 quotes an extinction power law near lambda**-3.48),
#: so its density is read at the red band, where oxygenated haemoglobin is nearly
#: transparent.
OD_MELANIN_AXIS = (1.0, 0.0, 0.0)
#: Haemoglobin axis in linear-sRGB optical-density space: the green Q-band
#: absorption (~542 nm) measured against the red band as a scattering baseline.
#: Normalised, i.e. ``(-1, +1, 0) / sqrt(2)``; the two axes are orthonormal, which
#: is what keeps the split numerically stable when only three broad bands exist.
OD_HEMOGLOBIN_AXIS = (-0.7071067811865476, 0.7071067811865476, 0.0)
#: Reference optical densities used to normalise the two indices into 0..1.
OD_MELANIN_REFERENCE = 1.7
OD_HEMOGLOBIN_REFERENCE = 0.85
#: Floor for the logarithm so that near-black pixels stay finite.
OD_LINEAR_FLOOR = 1e-4

# ---------------------------------------------------------------------------
# Illuminant estimation -- contract section 5B
# ---------------------------------------------------------------------------

#: Nominal CCT (kelvin) for each ``capture.illuminantGuess`` value.
ILLUMINANT_GUESS_CCT = {
    "daylight": 6500.0,
    "shade": 7500.0,
    "tungsten": 2856.0,
    "fluorescent": 4000.0,
    "led": 4500.0,
    "unknown": 6500.0,
}
#: Kinds that are physically closer to the CIE daylight locus than to Planck.
ILLUMINANT_GUESS_LOCUS = {
    "daylight": "daylight",
    "shade": "daylight",
    "tungsten": "planckian",
    "fluorescent": "planckian",
    "led": "planckian",
    "unknown": None,
}
#: Weight of the user prior when blending CCT (in mired) with the measurement.
#: Deliberately below 0.5: the prior nudges an uncertain estimate, it does not
#: override a measurement made from sclera / background pixels.
ILLUMINANT_PRIOR_WEIGHT = 0.25
ILLUMINANT_PRIOR_WEIGHT_D65_FALLBACK = 0.60
#: Search range for locus projection.
CCT_SEARCH_MIN_K = 1500.0
CCT_SEARCH_MAX_K = 20000.0
CCT_SEARCH_COARSE_STEPS = 190
CCT_SEARCH_REFINE_STEPS = 60

#: Sclera / teeth / background candidate selection.
SCLERA_MAX_SATURATION = 0.18
SCLERA_MIN_VALUE = 90
SCLERA_EYE_PATCH_SCALE = 0.35
SCLERA_MAX_PIXELS = 4000
TEETH_MIN_VALUE = 120
TEETH_MAX_SATURATION = 0.28
TEETH_MAX_PIXELS = 4000
#: Teeth are ranked by brightness (they are the brightest desaturated thing in
#: the mouth), and only the brightest quarter is kept so that a grey wall behind
#: the subject cannot outvote them.
TEETH_BRIGHT_QUANTILE = 0.25
HAAR_SMILE_CASCADE = "haarcascade_smile.xml"
AUTO_ROI_MOUTH_Y = (0.62, 0.92)
AUTO_ROI_MOUTH_X = (0.28, 0.72)
BACKGROUND_MAX_SATURATION = 0.35
BACKGROUND_MIN_VALUE = 40
BACKGROUND_EDGE_MARGIN_RATIO = 0.08
#: Fraction of the least-saturated border-band pixels kept as "the background is
#: the neutral thing here". A relative pick is required because a genuinely
#: neutral wall photographed under tungsten *is* chromatic; an absolute threshold
#: alone would reject every warm indoor capture.
BACKGROUND_NEUTRAL_QUANTILE = 0.25
#: Candidate pixel counts below which a source is discarded.
ILLUMINANT_MIN_CANDIDATE_PIXELS = 40
#: Default white point assumed when nothing at all can be estimated.
ASSUMED_D65_XY = (0.31271, 0.32902)

# ---------------------------------------------------------------------------
# Card self-calibration -- contract section 2.4
# ---------------------------------------------------------------------------

#: Largest chroma (CIELAB C*) tolerated across the neutral patches after the
#: printed paper's cast has been divided out.
CALIBRATION_GRAY_MAX_CHROMA = 6.0
#: Mean CIEDE2000 residual after calibration that still counts as "good".
CALIBRATION_GOOD_DELTA_E = 3.0
#: ... and as "fair"; anything worse is "poor".
CALIBRATION_FAIR_DELTA_E = 6.0
#: ``(min, max, steps)`` of the single global luminance gain searched during
#: calibration. Only one gain is allowed because the overall exposure is unknown.
CALIBRATION_LUMINANCE_SEARCH = (0.70, 1.40, 15)

# ---------------------------------------------------------------------------
# Advice -- contract section 6
# ---------------------------------------------------------------------------

ADVICE_HUE_OFFSET_MIN_DEG = 25.0
ADVICE_HUE_OFFSET_MAX_DEG = 60.0
ADVICE_CHROMA_RATIO_MIN = 0.8
ADVICE_CHROMA_RATIO_MAX = 1.6
ADVICE_LIGHTNESS_DELTA_TARGETS = (-30.0, -16.0, 16.0, 30.0)

ADVICE_AVOID_HUE_DELTA_MAX_DEG = 15.0
ADVICE_AVOID_CHROMA_RATIO_MIN = 1.6
ADVICE_AVOID_LIGHTNESS_DELTA_MAX = 6.0
ADVICE_AVOID_CHROMA_MAX = 12.0

#: Contrast level cut points on |L*_hair - L*_skin|.
CONTRAST_LEVEL_CUTS = (("high", 45.0), ("medium", 25.0))
CONTRAST_LEVEL_LOW = "low"

DISCLAIMER = (
    "本结果基于单张照片的相机响应估计，不能替代分光测色仪或专业色彩顾问。"
)

# ---------------------------------------------------------------------------
# Haar cascades (shipped with opencv-python-headless, zero download)
# ---------------------------------------------------------------------------

HAAR_FACE_CASCADE = "haarcascade_frontalface_default.xml"
HAAR_EYE_CASCADE = "haarcascade_eye.xml"
FACE_DETECT_SCALE_FACTOR = 1.1
FACE_DETECT_MIN_NEIGHBORS = 5
FACE_DETECT_MIN_SIZE_RATIO = 0.12
#: Auto-ROI geometry, as fractions of the detected face bounding box.
AUTO_ROI_JAW_Y = (0.55, 0.80)
AUTO_ROI_JAW_X = (0.18, 0.82)
AUTO_ROI_NECK_Y = (1.02, 1.25)
AUTO_ROI_NECK_X = (0.25, 0.75)

# ---------------------------------------------------------------------------
# Storage / privacy -- contract section 7
# ---------------------------------------------------------------------------

ARCHIVE_DIRNAME = "archive"
ARCHIVE_DATE_FORMAT = "%Y-%m"
DB_FILENAME = "skintone.sqlite3"
ARCHIVE_JPEG_QUALITY = 95
ARCHIVE_SUFFIX = ".jpg"
#: EXIF tags kept when stripping metadata before writing to disk.
EXIF_ORIENTATION_TAG = 274
EXIF_DATETIME_ORIGINAL_TAG = 36867
EXIF_DATETIME_TAG = 306
RESULT_ID_PREFIX_LENGTH = 26


# ---------------------------------------------------------------------------
# Runtime settings (.env)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime settings, read from ``.env`` / environment (prefix ``SKINTONE_``)."""

    model_config = SettingsConfigDict(
        env_prefix="SKINTONE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    allowed_origins: str = "*"
    rate_limit: str = "30/minute"
    #: 每日配额与防滥用闸门（**不是认证**）。刻意不使用客户端 IP：
    #: CGNAT 会让大量真实用户共用出口 IP，按 IP 限流必然误伤。
    quota_enabled: bool = True
    #: 单指纹每日次数（X-Client-Id）
    quota_per_day_client: int = 5
    #: 单指纹两次成功测量之间的冷却秒数
    quota_cooldown_seconds: float = 15.0
    #: 全站每日次数硬天花板——无论对方怎么换指纹都撞得到
    quota_global_per_day: int = 200
    #: 全站每日上传字节预算（MB）——直接保护磁盘
    quota_global_mb_per_day: float = 500.0
    #: 全站每分钟令牌桶（不依赖任何标识，给洪水限速）
    global_rate_limit: str = "120/minute"
    #: 同时进行的分析数上限；超出排队，保护 CPU
    max_concurrent_analyses: int = 2
    #: 排队等待上限（秒），超过就明确拒绝而不是无限等
    queue_wait_seconds: float = 20.0
    #: 配额库文件；留空则用 data_dir/guard.sqlite
    quota_db: str = ""
    max_upload_mb: int = 20
    data_dir: Path = Path("./data")
    retention_days: int = 30
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    #: Cloudflare named-tunnel settings consumed by ``scripts/tunnel.ps1``.
    tunnel_name: str = "skintone-api"
    tunnel_hostname: str = "skintest.0721.luxe"
    tunnel_token_file: str = ""
    spec_version: str = SPEC_VERSION
    server_version: str = SERVER_VERSION

    @property
    def origins(self) -> list[str]:
        """Return ``allowed_origins`` split into a CORS origin list."""
        parts = [item.strip() for item in self.allowed_origins.split(",")]
        return [item for item in parts if item]

    @property
    def auth_required(self) -> bool:
        """True when a non-empty API key was configured."""
        return bool(self.api_key)

    @property
    def quota_db_path(self) -> Path:
        """Guard-store path (defaults to ``data_dir/guard.sqlite``)."""
        return Path(self.quota_db) if self.quota_db else self.data_dir / "guard.sqlite"

    @property
    def max_upload_bytes(self) -> int:
        """Upload size ceiling in bytes, derived from ``max_upload_mb``."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def archive_dir(self) -> Path:
        """Directory holding the original-image archive (``data/archive``)."""
        return self.data_dir / ARCHIVE_DIRNAME

    @property
    def db_path(self) -> Path:
        """Path of the SQLite metadata database."""
        return self.data_dir / DB_FILENAME

    @property
    def rate_limit_capacity(self) -> int:
        """Burst capacity parsed from ``rate_limit`` (``"30/minute"`` -> 30)."""
        return _parse_rate_limit(self.rate_limit)[0]

    @property
    def rate_limit_per_second(self) -> float:
        """Refill rate in tokens per second parsed from ``rate_limit``."""
        return _parse_rate_limit(self.rate_limit)[1]

    @property
    def global_rate_limit_capacity(self) -> int:
        """Burst capacity for the site-wide limiter (identifier-free gate)."""
        return _parse_rate_limit(self.global_rate_limit)[0]

    @property
    def global_rate_limit_per_second(self) -> float:
        """Refill rate (tokens/second) for the site-wide limiter."""
        return _parse_rate_limit(self.global_rate_limit)[1]


def _parse_rate_limit(spec: str) -> tuple[int, float]:
    """Parse ``"30/minute"`` into ``(capacity, tokens_per_second)``.

    Recognised windows: ``second``, ``minute``, ``hour``, ``day``. An unparsable
    string falls back to 30/minute.
    """
    windows = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}
    try:
        count_text, _, window = spec.partition("/")
        capacity = int(count_text.strip())
        seconds = windows[window.strip().lower()]
        if capacity <= 0:
            raise ValueError
        return capacity, capacity / seconds
    except (ValueError, KeyError):
        return 30, 30 / 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()

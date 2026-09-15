"""Advice generation must stay inside the contract's numeric rules (section 6)."""

from __future__ import annotations

import numpy as np
import pytest

from skintone import config
from skintone.color import palette
from skintone.color.srgb import lab_to_lch

SKIN = np.array([62.14, 12.42, 18.91])


def test_palette_entries_have_the_contract_shape() -> None:
    """Every entry needs hex, lab and name."""
    for entry in palette.build_palette(SKIN):
        assert set(entry) == {"hex", "lab", "name"}
        assert isinstance(entry["hex"], str) and entry["hex"].startswith("#")
        assert len(entry["lab"]) == 3
        assert isinstance(entry["name"], str) and entry["name"]


def test_palette_is_inside_the_srgb_gamut() -> None:
    """A recommendation that cannot be displayed is useless."""
    for entry in palette.build_palette(SKIN):
        assert palette.round_trip_error(np.asarray(entry["lab"])) < 1.0


def test_palette_specs_stay_inside_the_contract_windows() -> None:
    """The generation rules themselves must obey section 6."""
    for offset, ratio, delta in palette._palette_specs(20.0):
        assert config.ADVICE_HUE_OFFSET_MIN_DEG <= abs(offset) <= config.ADVICE_HUE_OFFSET_MAX_DEG
        assert config.ADVICE_CHROMA_RATIO_MIN <= ratio <= config.ADVICE_CHROMA_RATIO_MAX
        assert delta != 0.0


def test_palette_hue_offsets_follow_the_harmony_rule() -> None:
    """Produced colours must keep a 25-60 degree offset from the skin hue."""
    _, _, skin_hue = lab_to_lch(SKIN)
    for entry in palette.build_palette(SKIN):
        _, _, hue = lab_to_lch(np.asarray(entry["lab"]))
        delta = abs(((hue - skin_hue) + 180.0) % 360.0 - 180.0)
        assert config.ADVICE_HUE_OFFSET_MIN_DEG - 0.5 <= delta <= config.ADVICE_HUE_OFFSET_MAX_DEG + 0.5


def test_palette_chroma_never_exceeds_the_ratio_cap() -> None:
    """Chroma is capped at 1.6x skin chroma; gamut clipping may only lower it."""
    _, skin_chroma, _ = lab_to_lch(SKIN)
    for entry in palette.build_palette(SKIN):
        _, chroma_value, _ = lab_to_lch(np.asarray(entry["lab"]))
        assert chroma_value <= skin_chroma * config.ADVICE_CHROMA_RATIO_MAX + 0.5
        assert chroma_value > 0.0


def test_palette_respects_the_limit() -> None:
    """The caller's limit must be honoured."""
    assert len(palette.build_palette(SKIN, limit=2)) == 2


def test_avoid_entries_explain_themselves() -> None:
    """A rejection must cite the numeric rule that produced it."""
    for entry in palette.build_avoid(SKIN):
        assert entry["hex"].startswith("#")
        assert entry["reason"]


def test_contrast_level_uses_the_hair_difference() -> None:
    """With a hair colour the level comes from |L_hair - L_skin|."""
    level, warning = palette.contrast_level(10.0, 62.14)
    assert level == "high"
    assert warning is None
    assert palette.contrast_level(35.0, 62.14)[0] == "medium"
    assert palette.contrast_level(55.0, 62.14)[0] == "low"


def test_contrast_level_without_hair_warns() -> None:
    """Without a hair colour the estimate is single-axis and must say so."""
    level, warning = palette.contrast_level(None, 62.14)
    assert level in {"high", "medium", "low"}
    assert warning is not None
    assert "对比度" in warning


def test_build_advice_returns_warnings_separately() -> None:
    """``build_advice`` must surface the caveat rather than bury it."""
    advice, warnings = palette.build_advice(SKIN, None)
    assert set(advice) == {"contrastLevel", "palette", "avoid"}
    assert warnings and "对比度" in warnings[0]
    advice_with_hair, warnings_with_hair = palette.build_advice(SKIN, 10.0)
    assert advice_with_hair["contrastLevel"] == "high"
    assert warnings_with_hair == []


def test_named_colours_are_not_empty_for_extreme_lightness() -> None:
    """Very light and very dark skin must still produce sensible names."""
    for lab in ([92.0, 2.0, 8.0], [18.0, 8.0, 10.0], [50.0, 0.5, 0.5]):
        for entry in palette.build_palette(np.asarray(lab)):
            assert entry["name"]


def test_gamut_clipping_preserves_hue() -> None:
    """Out-of-gamut colours must be pulled in by chroma only."""
    wild = np.array([55.0, 90.0, -80.0])
    clipped = palette.clip_to_gamut(wild)
    assert clipped[0] == pytest.approx(wild[0], abs=1e-9)
    _, _, hue_before = lab_to_lch(wild)
    _, _, hue_after = lab_to_lch(clipped)
    assert hue_after == pytest.approx(hue_before, abs=1e-6)

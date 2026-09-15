"""The card definition must match contract section 3 exactly."""

from __future__ import annotations

import pytest

from skintone import config
from skintone.cards import definitions


def test_card_ids() -> None:
    """Exactly one card is served, with the contract's id."""
    assert definitions.card_ids() == ["skintone-a4-v1"]


def test_paper_geometry() -> None:
    """A4 portrait at 300 DPI."""
    spec = definitions.card_spec(config.CARD_ID)
    assert spec is not None
    assert spec["paper"] == {"name": "A4", "widthMm": 210.0, "heightMm": 297.0, "dpi": 300}
    assert (config.CARD_WIDTH_PX, config.CARD_HEIGHT_PX) == (2480, 3508)


def test_marker_layout() -> None:
    """Four DICT_4X4_50 markers, 20 mm, at the contract's centres."""
    assert config.ARUCO_DICTIONARY == "DICT_4X4_50"
    assert [marker.id for marker in definitions.MARKERS] == [0, 1, 2, 3]
    assert [marker.center_mm for marker in definitions.MARKERS] == [
        (25.0, 25.0),
        (185.0, 25.0),
        (25.0, 272.0),
        (185.0, 272.0),
    ]
    assert all(marker.size_mm == 20.0 for marker in definitions.MARKERS)


def test_gray_ramp_values() -> None:
    """The gray ramp must carry the contract's five steps plus the K05 anchor."""
    expected = {
        "G90": ((45.0, 65.0), (245, 245, 245)),
        "G70": ((77.0, 65.0), (200, 200, 200)),
        "G50": ((109.0, 65.0), (160, 160, 160)),
        "G30": ((141.0, 65.0), (120, 120, 120)),
        "G10": ((173.0, 65.0), (75, 75, 75)),
        "K05": ((25.0, 215.0), (40, 40, 40)),
    }
    for patch_id, (center, srgb) in expected.items():
        patch = definitions.get_patch(patch_id)
        assert patch is not None, patch_id
        assert patch.kind == "gray"
        assert patch.center_mm == center
        assert patch.nominal_srgb == srgb
        assert patch.size_mm == 26.0


def test_color_grid_layout_and_values() -> None:
    """24 colour patches at the contract's column/row centres."""
    grid = [patch for patch in definitions.PATCHES if patch.size_mm == 22.0]
    assert len(grid) == 24
    assert [(p.center_mm[0], p.center_mm[1]) for p in grid[:6]] == [
        (31.0, 101.0),
        (61.0, 101.0),
        (91.0, 101.0),
        (121.0, 101.0),
        (151.0, 101.0),
        (181.0, 101.0),
    ]
    assert grid[6].center_mm == (31.0, 131.0)
    assert grid[18].center_mm == (31.0, 191.0)


def test_row_four_hex_values() -> None:
    """Spot-check the last row against the contract's hex table."""
    expected = {
        "W": (255, 255, 255),
        "K": (0, 0, 0),
        "G90b": (245, 245, 245),
        "G50b": (160, 160, 160),
        "P1": (232, 160, 180),
        "T1": (47, 79, 111),
    }
    for patch_id, srgb in expected.items():
        patch = definitions.get_patch(patch_id)
        assert patch is not None
        assert patch.nominal_srgb == srgb


def test_neutral_patches_are_chroma_free() -> None:
    """Every ``kind=gray`` patch must have exactly zero chromaticity."""
    for patch in definitions.PATCHES:
        if patch.kind == "gray":
            assert patch.nominal_srgb[0] == patch.nominal_srgb[1] == patch.nominal_srgb[2]


def test_kind_assignment() -> None:
    """Skin blocks are ``skin``, olive blocks ``olive``, hue blocks ``color``."""
    assert definitions.get_patch("SK4").kind == "skin"
    assert definitions.get_patch("OL1").kind == "olive"
    assert definitions.get_patch("R").kind == "color"
    assert definitions.get_patch("T1").kind == "color"


def test_sample_bounds_are_the_inner_sixty_percent() -> None:
    """Sampling must use the central 60 % of each patch, in canonical pixels."""
    patch = definitions.get_patch("SK3")
    assert patch is not None
    x0, y0, x1, y1 = definitions.patch_sample_bounds_px(patch)
    expected_side = 22.0 * config.CARD_PX_PER_MM * 0.60
    assert (x1 - x0) == pytest.approx(expected_side, abs=1)
    assert (y1 - y0) == pytest.approx(expected_side, abs=1)
    center_x = 91.0 * config.CARD_PX_PER_MM
    assert (x0 + x1) / 2 == pytest.approx(center_x, abs=1)


def test_every_patch_is_inside_the_paper() -> None:
    """No patch may hang off the sheet."""
    for patch in definitions.PATCHES:
        x0, y0, x1, y1 = definitions.patch_sample_bounds_px(patch)
        assert 0 <= x0 < x1 <= config.CARD_WIDTH_PX
        assert 0 <= y0 < y1 <= config.CARD_HEIGHT_PX


def test_marker_corners_order_is_aruco_order() -> None:
    """Corners must run TL, TR, BR, BL so they pair with ``cv2.aruco`` output."""
    marker = definitions.MARKERS[0]
    corners = definitions.marker_corners_mm(marker)
    assert corners[0] == (15.0, 15.0)
    assert corners[1] == (35.0, 15.0)
    assert corners[2] == (35.0, 35.0)
    assert corners[3] == (15.0, 35.0)


def test_unknown_card_returns_none() -> None:
    """An unknown card id must be reported, not guessed."""
    assert definitions.card_spec("nope") is None
    assert definitions.get_patch("nope") is None


def test_card_spec_contains_every_patch_and_instruction() -> None:
    """The printable spec must expose all 30 patches and the printing rules."""
    spec = definitions.card_spec(config.CARD_ID)
    assert spec is not None
    assert len(spec["patches"]) == len(definitions.PATCHES) == 30
    assert spec["specVersion"] == "1.0.0"
    assert any("100%" in line for line in spec["instructions"])

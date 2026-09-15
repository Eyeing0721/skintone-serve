"""skintone-serve: reproducible skin-tone measurement backend.

Contract: ``CONTRACT.md`` v1.0.0 (cross-repository single source of truth).

Everything the service computes lives under :mod:`skintone.color` (hand written,
auditable colour science) and :mod:`skintone.vision` (OpenCV measurement code).
No third-party colour-science library is used anywhere.

Units and colour spaces used in this package (see ``docs/ALGORITHM.md``):

* ``srgb``      -- 8-bit or 0..1 *encoded* sRGB, IEC 61966-2-1 transfer function.
* ``linear``    -- 0..1 *linear* sRGB primaries, no transfer function applied.
* ``xyz``       -- CIE 1931 2-degree tristimulus, D65 white unless stated.
* ``lab``       -- CIELAB with D65 white point (Xn=0.95047, Yn=1.0, Zn=1.08883).
* ``xy``        -- CIE 1931 chromaticity, invariant to luminance.
* ``uv``        -- CIE 1960 UCS, ``u = 4X/(X+15Y+3Z)``, ``v = 6Y/(X+15Y+3Z)``.
* ``cct``       -- correlated colour temperature in kelvin.
* ``duv``       -- signed distance from a chromaticity to a locus, in CIE 1960
                   ``uv``; > 0 means above the locus (greenish), < 0 below (pinkish).
* ``mm``        -- millimetres on the printed reference card.
* ``ita``       -- Individual Typology Angle, degrees.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

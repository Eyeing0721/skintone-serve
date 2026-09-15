"""Hand-written, auditable colour science (contract section 4).

No third-party colour library is used: every conversion and every difference in
this package is implemented from its published definition so that a reviewer can
trace a number in an API response back to a formula.
"""

from __future__ import annotations

__all__ = ["adaptation", "ciede2000", "illuminant", "palette", "skin", "srgb"]

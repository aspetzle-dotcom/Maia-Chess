"""Clock labels are estimates of server-recorded consumption, not cognition."""

import math
import re


def time_control(value: str) -> tuple[float, float] | None:
    """Support a single sudden-death/Fischer control; reject staged/daily controls."""
    if not re.fullmatch(r"\d+(?:\.\d+)?(?:\+\d+(?:\.\d+)?)?", value):
        return None
    parts = value.split("+")
    base, increment = float(parts[0]), float(parts[1]) if len(parts) == 2 else 0.0
    return (base, increment) if base > 0 else None


def elapsed_seconds(before, after, increment: float, *, first_move=False):
    """Mask first moves, missing clocks and inconsistent clock transitions.

    A missing annotation must invalidate the next same-side difference. Do not
    clamp negative differences to zero: that would turn bad labels into premoves.
    """
    if first_move or before is None or after is None:
        return None
    if not all(math.isfinite(x) for x in (before, after, increment)):
        return None
    if before < 0 or after < 0 or increment < 0:
        return None
    elapsed = before + increment - after
    if elapsed < -1e-6 or elapsed > before + 1e-6:
        return None
    return min(before, max(0.0, elapsed))

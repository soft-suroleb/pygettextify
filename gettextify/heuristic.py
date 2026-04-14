"""Deterministic rule-based baseline classifier.

Used as a reference point to compare against the ML (CatBoost) model.
Returns the same three labels as the ML predictor: "IN", "OUT", "GRAY".
"""

from __future__ import annotations

from .features import (
    STOPLIST,
    _is_camel_case,
    _is_json,
    _is_snake_case,
    _is_xml,
    _looks_like_path,
)

LABEL_IN = "IN"
LABEL_OUT = "OUT"
LABEL_GRAY = "GRAY"


def classify_heuristic(key: str, with_format: bool = False) -> str:
    """Classify a string literal using deterministic rules.

    Rules are applied in priority order (first match wins):

    OUT rules — strong signals that the string is NOT user-visible:
      1. Empty or whitespace-only string.
      2. All-uppercase string (constant / enum value).
      3. snake_case or CamelCase identifier.
      4. Known technical keyword (STOPLIST).
      5. Looks like JSON, XML, or a file-system path.

    IN rules — strong signals that the string IS user-visible:
      6. Starts with an uppercase letter AND contains a space (natural language phrase).
      7. Ends with sentence-ending punctuation (., !, ?).
      8. Contains a format placeholder and is non-trivial (≥2 words).

    GRAY — everything else that cannot be decided with confidence.
    """
    stripped = key.strip()

    # --- OUT rules ---
    if not stripped:
        return LABEL_OUT

    if key.isupper() and len(key) <= 30:
        return LABEL_OUT

    if _is_snake_case(key) or _is_camel_case(key):
        return LABEL_OUT

    if stripped.upper() in STOPLIST:
        return LABEL_OUT

    if _is_json(key) or _is_xml(key) or _looks_like_path(key):
        return LABEL_OUT

    # --- IN rules ---
    if key[0].isupper() and ' ' in key:
        return LABEL_IN

    if stripped[-1] in '.!?':
        return LABEL_IN

    if with_format and len(key.split()) >= 2:
        return LABEL_IN

    return LABEL_GRAY


def classify_many(
    strings: list[tuple[str, bool]],
) -> list[str]:
    """Classify a list of (key, with_format) pairs.

    Args:
        strings: list of (key, with_format) tuples.

    Returns:
        List of label strings ("IN", "OUT", "GRAY").
    """
    return [classify_heuristic(key, wf) for key, wf in strings]

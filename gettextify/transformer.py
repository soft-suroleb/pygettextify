"""
Source code transformation:
  - wrapping string literals in _()
  - marking gray-zone literals with a review comment
  - adding gettext import (if needed)
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import StringLiteral

GRAY_COMMENT = "  # i18n: review"


def wrap_strings(source: str, literals: list[StringLiteral]) -> str:
    """Inserts _( ... ) around the specified string literals."""
    if not literals:
        return source

    lines = source.splitlines(keepends=True)

    # process bottom-up / right-to-left so offsets don't shift
    sorted_lits = sorted(
        literals,
        key=lambda l: (l.end_lineno, l.end_col_offset),
        reverse=True,
    )

    for lit in sorted_lits:
        if lit.lineno == lit.end_lineno:
            idx = lit.lineno - 1
            line = lines[idx]
            lines[idx] = (
                line[:lit.col_offset]
                + "_(" + line[lit.col_offset:lit.end_col_offset] + ")"
                + line[lit.end_col_offset:]
            )
        else:
            # multiline string
            end_idx = lit.end_lineno - 1
            end_line = lines[end_idx]
            lines[end_idx] = (
                end_line[:lit.end_col_offset]
                + ")"
                + end_line[lit.end_col_offset:]
            )

            start_idx = lit.lineno - 1
            start_line = lines[start_idx]
            lines[start_idx] = (
                start_line[:lit.col_offset]
                + "_("
                + start_line[lit.col_offset:]
            )

    return "".join(lines)


def mark_gray_strings(source: str, literals: list[StringLiteral]) -> str:
    """Appends ``# i18n: review`` comment to lines containing gray literals.

    Each affected line gets the comment appended at most once, even if
    multiple gray literals appear on the same line.
    """
    if not literals:
        return source

    lines = source.splitlines(keepends=True)

    # Collect the set of 0-based line indices that contain a gray literal.
    gray_line_indices: set[int] = set()
    for lit in literals:
        for lineno in range(lit.lineno, lit.end_lineno + 1):
            gray_line_indices.add(lineno - 1)

    for idx in sorted(gray_line_indices):
        line = lines[idx]
        if GRAY_COMMENT.strip() not in line:
            # Strip trailing newline, append comment, restore newline.
            stripped = line.rstrip("\n")
            newline = line[len(stripped):]
            lines[idx] = stripped + GRAY_COMMENT + newline

    return "".join(lines)


def add_gettext_import(source: str) -> str:
    """Adds ``from gettext import gettext as _`` if _ is not already defined."""
    if _has_gettext_setup(source):
        return source

    import_line = "from gettext import gettext as _\n"
    lines = source.splitlines(keepends=True)

    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_idx = i + 1

    if insert_idx > 0:
        lines.insert(insert_idx, import_line)
    else:
        lines.insert(0, import_line)

    return "".join(lines)


def _has_gettext_setup(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in (
                    'gettext', 'gettext_lazy',
                    'ugettext', 'ugettext_lazy',
                ):
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if 'gettext' in (alias.name or ''):
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '_':
                    return True
    return False

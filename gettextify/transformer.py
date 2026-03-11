"""
Трансформация исходного кода:
  - оборачивание строковых литералов в _()
  - добавление импорта gettext (если необходимо)
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import StringLiteral


def wrap_strings(source: str, literals: list[StringLiteral]) -> str:
    """Вставляет _( … ) вокруг указанных строковых литералов."""
    if not literals:
        return source

    lines = source.splitlines(keepends=True)

    # обрабатываем снизу-вверх / справа-налево, чтобы не сбивались смещения
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
            # многострочная строка
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


def add_gettext_import(source: str) -> str:
    """Добавляет ``from gettext import gettext as _``, если _ ещё не определён."""
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

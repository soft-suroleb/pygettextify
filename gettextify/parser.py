"""
Извлечение строковых литералов из Python-файла с помощью AST.

Для каждого литерала определяется:
  - позиция в исходном коде (line/col)
  - используется ли с .format() или оператором %
  - является ли docstring-ом
  - обёрнут ли уже в _()
"""

import ast
from dataclasses import dataclass


@dataclass
class StringLiteral:
    value: str
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    with_format: bool = False
    is_docstring: bool = False
    is_wrapped: bool = False


def extract_strings(source: str) -> list[StringLiteral]:
    tree = ast.parse(source)

    format_ids: set[int] = set()
    mod_ids: set[int] = set()
    docstring_ids: set[int] = set()
    wrapped_ids: set[int] = set()

    for node in ast.walk(tree):
        # --- docstrings (первый Expr(Constant(str)) в теле) ---
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstring_ids.add(id(body[0].value))

        # --- "...".format(...) ---
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr == 'format'
                    and isinstance(func.value, ast.Constant)
                    and isinstance(func.value.value, str)):
                format_ids.add(id(func.value))

            # уже обёрнуто в _()
            if isinstance(func, ast.Name) and func.id == '_':
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        wrapped_ids.add(id(arg))

        # --- "..." % ... ---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                mod_ids.add(id(node.left))

    # --- собираем литералы ---
    literals: list[StringLiteral] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        nid = id(node)
        literals.append(StringLiteral(
            value=node.value,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            with_format=(nid in format_ids or nid in mod_ids),
            is_docstring=(nid in docstring_ids),
            is_wrapped=(nid in wrapped_ids),
        ))

    return literals

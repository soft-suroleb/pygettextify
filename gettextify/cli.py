"""
gettextify CLI.

Usage:
    python -m gettextify <path> [--threshold 0.78] [--in-place] [--verbose]

    <path> — a .py file or directory.
    If a directory is given, it is traversed recursively and all
    .py files found are processed (implies --inplace).

Pipeline:
    1) AST parsing of the file -> extraction of string literals
    2) Feature computation for each literal
    3) CatBoost prediction — whether to wrap in _()
    4) Source code transformation + adding import gettext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .features import compute_features
from .parser import extract_strings
from .predictor import (
    DEFAULT_THRESHOLD_IN,
    DEFAULT_THRESHOLD_OUT,
    LABEL_GRAY,
    LABEL_IN,
    MODEL_PATH,
    load_model,
    predict,
)
from .transformer import add_gettext_import, mark_gray_strings, wrap_strings


def _collect_py_files(path: Path) -> list[Path]:
    """Recursively collects all .py files from a directory."""
    return sorted(path.rglob("*.py"))


def process(
    filepath: str,
    *,
    model_path: str = MODEL_PATH,
    threshold: float | None = None,
    threshold_in: float = DEFAULT_THRESHOLD_IN,
    threshold_out: float = DEFAULT_THRESHOLD_OUT,
    inplace: bool = False,
    verbose: bool = False,
):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    literals = extract_strings(source)
    candidates = [
        lit for lit in literals
        if not lit.is_docstring and not lit.is_wrapped and not lit.is_fstring_part
    ]

    if not candidates:
        if verbose:
            print(f"  {filepath}: no candidates found, skipping.")
        return

    if verbose:
        print(f"  Found literals: {len(literals)}, candidates: {len(candidates)}")

    value_counts: dict[str, int] = {}
    for lit in candidates:
        value_counts[lit.value] = value_counts.get(lit.value, 0) + 1

    features_list = [
        compute_features(
            key=lit.value,
            with_format=lit.with_format,
            count=value_counts[lit.value] + 1,
        )
        for lit in candidates
    ]

    model = load_model(model_path)
    # threshold shorthand: overrides both threshold_in and threshold_out
    if threshold is not None:
        threshold_in = threshold
        threshold_out = threshold
    labels = predict(model, features_list, threshold_in=threshold_in, threshold_out=threshold_out)

    to_wrap = [lit for lit, lbl in zip(candidates, labels) if lbl == LABEL_IN]
    to_gray = [lit for lit, lbl in zip(candidates, labels) if lbl == LABEL_GRAY]

    if verbose:
        print(f"  IN (to wrap): {len(to_wrap)}, GRAY (to review): {len(to_gray)}")
        for lit in to_wrap:
            print(f"    [IN]   line {lit.lineno}: {lit.value!r}")
        for lit in to_gray:
            print(f"    [GRAY] line {lit.lineno}: {lit.value!r}")

    if not to_wrap and not to_gray:
        if verbose:
            print(f"  {filepath}: model found no strings to internationalize.")
        return

    new_source = wrap_strings(source, to_wrap)
    new_source = mark_gray_strings(new_source, to_gray)
    if to_wrap:
        new_source = add_gettext_import(new_source)

    if inplace:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)
        print(
            f"File {filepath} modified: wrapped {len(to_wrap)} strings,"
            f" marked {len(to_gray)} for review."
        )
    else:
        sys.stdout.write(new_source)


def process_directory(
    dirpath: str,
    *,
    model_path: str = MODEL_PATH,
    threshold: float | None = None,
    threshold_in: float = DEFAULT_THRESHOLD_IN,
    threshold_out: float = DEFAULT_THRESHOLD_OUT,
    verbose: bool = False,
):
    """Recursively processes all .py files in a directory (always in-place)."""
    py_files = _collect_py_files(Path(dirpath))

    if not py_files:
        print(f"No .py files found in directory {dirpath}.")
        return

    print(f"Found .py files: {len(py_files)}")

    processed = 0
    for py_file in py_files:
        if verbose:
            print(f"\n--- {py_file} ---")
        try:
            process(
                str(py_file),
                model_path=model_path,
                threshold=threshold,
                threshold_in=threshold_in,
                threshold_out=threshold_out,
                inplace=True,
                verbose=verbose,
            )
            processed += 1
        except Exception as exc:
            print(f"Error processing {py_file}: {exc}", file=sys.stderr)

    print(f"\nProcessed files: {processed}/{len(py_files)}")


def main():
    p = argparse.ArgumentParser(
        prog="gettextify",
        description="Automatic internationalization of Python string literals",
    )
    p.add_argument(
        "path",
        help="Path to a .py file or directory (recursive traversal)",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Shorthand: sets both --threshold-in and --threshold-out to this value",
    )
    p.add_argument(
        "--threshold-in", type=float, default=DEFAULT_THRESHOLD_IN,
        dest="threshold_in",
        help=f"P(IN) >= threshold → wrap in _()  (default {DEFAULT_THRESHOLD_IN})",
    )
    p.add_argument(
        "--threshold-out", type=float, default=DEFAULT_THRESHOLD_OUT,
        dest="threshold_out",
        help=f"P(IN) < 1-threshold → skip; else mark as GRAY  (default {DEFAULT_THRESHOLD_OUT})",
    )
    p.add_argument(
        "--model", default=MODEL_PATH,
        help="Path to the model file (.cbm)",
    )
    p.add_argument(
        "--inplace", action="store_true",
        help="Modify file in place (otherwise output to stdout)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = p.parse_args()

    target = Path(args.path)

    if target.is_dir():
        process_directory(
            str(target),
            model_path=args.model,
            threshold=args.threshold,
            threshold_in=args.threshold_in,
            threshold_out=args.threshold_out,
            verbose=args.verbose,
        )
    elif target.is_file():
        process(
            str(target),
            model_path=args.model,
            threshold=args.threshold,
            threshold_in=args.threshold_in,
            threshold_out=args.threshold_out,
            inplace=args.inplace,
            verbose=args.verbose,
        )
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

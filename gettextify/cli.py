"""
gettextify CLI.

Usage
-----
gettextify <path> [--threshold 0.78] [--inplace] [--verbose]
    Wrap translatable strings in _() and annotate gray-zone strings with
    ``# i18n: review``.  <path> may be a .py file or a directory
    (recursive; always in-place for directories).

gettextify scan <path>
    Scan a .py file or directory recursively and print every line that
    contains the gray-zone marker ``# i18n: review``.

Pipeline (mark):
    1) AST parsing → extraction of string literals
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
from .transformer import GRAY_COMMENT, add_gettext_import, mark_gray_strings, wrap_strings

GRAY_MARKER = GRAY_COMMENT.strip()  # "# i18n: review"


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


def scan_file(filepath: Path) -> list[tuple[int, str]]:
    """Returns a list of (lineno, line) pairs that contain the gray-zone marker."""
    try:
        lines = filepath.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [
        (i + 1, line)
        for i, line in enumerate(lines)
        if GRAY_MARKER in line
    ]


def scan_path(target: Path) -> None:
    """Scans a file or directory and prints all lines with the gray-zone comment."""
    if target.is_file():
        py_files = [target]
    else:
        py_files = _collect_py_files(target)
        if not py_files:
            print(f"No .py files found in {target}")
            return

    total = 0
    for py_file in py_files:
        matches = scan_file(py_file)
        for lineno, line in matches:
            print(f"{py_file}:{lineno}: {line}")
            total += 1

    if total == 0:
        print("No gray-zone comments found.")
    else:
        print(f"\nTotal: {total} line(s) with '{GRAY_MARKER}'")


def _build_scan_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gettextify scan",
        description=f"Find all lines marked with '{GRAY_MARKER}' (gray-zone review comment)",
    )
    p.add_argument("path", help="Path to a .py file or directory to scan")
    return p


def _build_mark_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gettextify",
        description="Automatic internationalization of Python string literals",
        epilog="Use 'gettextify scan <path>' to find existing gray-zone comments.",
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
    return p


def main():
    # Dispatch to scan subcommand when the first argument is "scan".
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        args = _build_scan_parser().parse_args(sys.argv[2:])
        target = Path(args.path)
        if not target.exists():
            print(f"Path not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        scan_path(target)
        return

    args = _build_mark_parser().parse_args()
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

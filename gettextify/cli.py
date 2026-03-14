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

import argparse
import sys
from pathlib import Path

from .features import compute_features
from .parser import extract_strings
from .predictor import DEFAULT_THRESHOLD, MODEL_PATH, load_model, predict
from .transformer import add_gettext_import, wrap_strings


def _collect_py_files(path: Path) -> list[Path]:
    """Recursively collects all .py files from a directory."""
    return sorted(path.rglob("*.py"))


def process(
    filepath: str,
    *,
    model_path: str = MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
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
    predictions = predict(model, features_list, threshold)

    to_wrap = [lit for lit, pred in zip(candidates, predictions) if pred]

    if verbose:
        print(f"  Literals to wrap: {len(to_wrap)}")
        for lit in to_wrap:
            print(f"    line {lit.lineno}: {lit.value!r}")

    if not to_wrap:
        if verbose:
            print(f"  {filepath}: model found no strings to internationalize.")
        return

    new_source = wrap_strings(source, to_wrap)
    new_source = add_gettext_import(new_source)

    if inplace:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)
        print(f"File {filepath} modified, wrapped {len(to_wrap)} strings.")
    else:
        sys.stdout.write(new_source)


def process_directory(
    dirpath: str,
    *,
    model_path: str = MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
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
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Probability threshold (default {DEFAULT_THRESHOLD})",
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
            verbose=args.verbose,
        )
    elif target.is_file():
        process(
            str(target),
            model_path=args.model,
            threshold=args.threshold,
            inplace=args.inplace,
            verbose=args.verbose,
        )
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

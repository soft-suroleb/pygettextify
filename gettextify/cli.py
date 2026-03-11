"""
CLI утилиты gettextify.

Использование:
    python -m gettextify <path> [--threshold 0.78] [--in-place] [--verbose]

    <path> — .py-файл или директория.
    Если передана директория, она обходится рекурсивно и обрабатываются
    все найденные .py-файлы (подразумевается --inplace).

Пайплайн:
    1) AST-парсинг файла → извлечение строковых литералов
    2) Вычисление фичей для каждого литерала
    3) Предсказание CatBoost — нужно ли оборачивать в _()
    4) Трансформация исходного кода + добавление import gettext
"""

import argparse
import sys
from pathlib import Path

from .features import compute_features
from .parser import extract_strings
from .predictor import DEFAULT_THRESHOLD, MODEL_PATH, load_model, predict
from .transformer import add_gettext_import, wrap_strings


def _collect_py_files(path: Path) -> list[Path]:
    """Рекурсивно собирает все .py-файлы из директории."""
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
        if not lit.is_docstring and not lit.is_wrapped
    ]

    if not candidates:
        if verbose:
            print(f"  {filepath}: кандидатов не найдено, пропуск.")
        return

    if verbose:
        print(f"  Найдено литералов: {len(literals)}, кандидатов: {len(candidates)}")

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
        print(f"  Литералов для обёртки: {len(to_wrap)}")
        for lit in to_wrap:
            print(f"    строка {lit.lineno}: {lit.value!r}")

    if not to_wrap:
        if verbose:
            print(f"  {filepath}: модель не нашла строк для интернационализации.")
        return

    new_source = wrap_strings(source, to_wrap)
    new_source = add_gettext_import(new_source)

    if inplace:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)
        print(f"Файл {filepath} изменён, обёрнуто строк: {len(to_wrap)}")
    else:
        sys.stdout.write(new_source)


def process_directory(
    dirpath: str,
    *,
    model_path: str = MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = False,
):
    """Рекурсивно обрабатывает все .py-файлы в директории (всегда inplace)."""
    py_files = _collect_py_files(Path(dirpath))

    if not py_files:
        print(f"В директории {dirpath} не найдено .py-файлов.")
        return

    print(f"Найдено .py-файлов: {len(py_files)}")

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
            print(f"Ошибка при обработке {py_file}: {exc}", file=sys.stderr)

    print(f"\nОбработано файлов: {processed}/{len(py_files)}")


def main():
    p = argparse.ArgumentParser(
        prog="gettextify",
        description="Автоматическая интернационализация строковых литералов Python",
    )
    p.add_argument(
        "path",
        help="Путь к .py-файлу или директории (рекурсивный обход)",
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Порог вероятности (default {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--model", default=MODEL_PATH,
        help="Путь к файлу модели (.cbm)",
    )
    p.add_argument(
        "--inplace", action="store_true",
        help="Изменить файл на месте (без этого — вывод в stdout)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Подробный вывод",
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
        print(f"Путь не найден: {args.path}", file=sys.stderr)
        sys.exit(1)

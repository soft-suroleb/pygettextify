"""Evaluate the heuristic (rule-based) baseline on the same test split used for ML.

Usage:
    python evaluate_baseline.py --dataset dataset.json

The script replicates the exact train/test split from train.py (seed=42, test_size=0.2)
so the results are directly comparable with the ML model metrics.

Output: confusion matrix + Precision / Recall / F1 for both the heuristic and
the ML model (at threshold 0.73).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from gettextify.features import FEATURE_NAMES, compute_features
from gettextify.heuristic import LABEL_IN, classify_heuristic
from gettextify.predictor import (
    DEFAULT_THRESHOLD_IN,
    DEFAULT_THRESHOLD_OUT,
    load_model,
    predict,
)

# ---------------------------------------------------------------------------
# Reuse dataset parsing from train.py (kept minimal to avoid import coupling)
# ---------------------------------------------------------------------------

def _convert_py2_to_py3(code: str) -> str:
    code = re.sub(r'(?m)^([ \t]*)print\s*$', r'\1print()', code)
    code = re.sub(
        r'(?m)^([ \t]*)print\s+(?!\()(.+)$',
        lambda m: f'{m.group(1)}print({m.group(2).strip()})',
        code,
    )
    code = re.sub(
        r'(?m)^([ \t]*)except\s+([a-zA-Z_][\w\.]*)(\s*,\s*([a-zA-Z_][\w]*))\s*:',
        lambda m: f'{m.group(1)}except {m.group(2)} as {m.group(4)}:',
        code,
    )
    code = re.sub(r'(?<![\w.])0([0-7]+)\b', r'0o\1', code)
    code = re.sub(r'(?<![\w.])0([1-9]\d*)\b', r'\1', code)
    return code


class _GettextError(Exception):
    pass


class _GettextFinder(ast.NodeVisitor):
    DEFAULT_GETTEXT_FUNC = ['gettext', 'gettext_lazy']
    GETTEXT_MODULE_FUNC = (
        'dgettext', 'dngettext', 'gettext', 'lgettext', 'ldgettext',
        'ldngettext', 'lngettext', 'ngettext', 'pgettext', 'dpgettext',
        'npgettext', 'dnpgettext', 'ugettext',
    )
    GETTEXT_FUNC_ARG_NUMBER = {'dgettext': 1}

    def __init__(self):
        self.func_names: set[str] = set()
        self.module_aliases: set[str] = set()
        self.dataset: dict[str, dict] = {}
        self.formats: list = []
        self.mods: list = []
        self.language_vars: set[str] = set()

    def _check_gettext(self, word):
        return word in self.DEFAULT_GETTEXT_FUNC or 'gettext' in word

    def _is_gt_name(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.func_names
        )

    def _is_gt_attr(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.module_aliases
            and node.func.attr in self.GETTEXT_MODULE_FUNC
        )

    def _append(self, node, *, target=False):
        with_inner = False
        if isinstance(node, ast.Constant):
            value = node.value
        elif (
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
        ):
            value = node.left.value
            with_inner = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Constant)
            and node.func.attr == 'format'
        ):
            value = node.func.value.value
            with_inner = True
        elif (
            isinstance(node, ast.JoinedStr)
            and len(node.values) > 1
            and isinstance(node.values[0], ast.Constant)
        ):
            value = node.values[0].value
            with_inner = True
        else:
            raise _GettextError()

        if value not in self.dataset:
            self.dataset[value] = {
                'value': value, 'count': 0,
                'with_format': False, 'with_inner_format': with_inner,
                'target': target,
            }
        self.dataset[value]['count'] += 1
        if target and not self.dataset[value]['target']:
            self.dataset[value]['target'] = True

    def visit_Import(self, node):
        for imp in node.names:
            if self._check_gettext(imp.name):
                self.module_aliases.add(imp.asname or imp.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for imp in node.names:
            if self._check_gettext(imp.name):
                self.func_names.add(imp.asname or imp.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and self._is_gt_name(node) and node.args:
            arg_no = self.GETTEXT_FUNC_ARG_NUMBER.get(node.func.id, 0)
            self._append(node.args[arg_no], target=True)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == 'format':
                self.formats.append(node)
            elif self._is_gt_attr(node) and node.args:
                self._append(node.args[0], target=True)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr in self.GETTEXT_MODULE_FUNC
        ):
            if isinstance(node.targets[0], ast.Name):
                self.func_names.add(node.targets[0].id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mod):
            self.mods.append(node)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            try:
                self._append(node, target=False)
            except _GettextError:
                pass
        self.generic_visit(node)

    def _process_format(self, node):
        base = node.func.value
        if self._is_gt_name(base) and base.args and isinstance(base.args[0], ast.Constant):
            self.dataset[base.args[0].value]['with_format'] = True

    def _process_mod(self, node):
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Call)
            and self._is_gt_name(node.left)
            and node.left.args
            and isinstance(node.left.args[0], ast.Constant)
        ):
            self.dataset[node.left.args[0].value]['with_format'] = True

    def run(self, tree):
        self.visit(tree)
        for node in self.formats:
            try:
                self._process_format(node)
            except Exception:
                pass
        for node in self.mods:
            try:
                self._process_mod(node)
            except Exception:
                pass


def _build_dataframe(raw_dataset: list[dict]) -> pd.DataFrame:
    rows = []
    global_index: dict[str, int] = {}
    for i, item in enumerate(raw_dataset):
        source = item['file']
        try:
            tree = ast.parse(source)
        except SyntaxError:
            try:
                tree = ast.parse(_convert_py2_to_py3(source))
            except SyntaxError as e:
                print(f'[{i}] SyntaxError, skipping: {e}', file=sys.stderr)
                continue

        finder = _GettextFinder()
        try:
            finder.run(tree)
        except _GettextError as e:
            print(f'[{i}] GettextError, skipping: {e}', file=sys.stderr)
            continue

        for value, raw in finder.dataset.items():
            global_index[value] = global_index.get(value, 0) + 1
            feat = compute_features(
                key=raw['value'],
                with_format=raw['with_format'],
                count=raw['count'],
                global_count=global_index[value],
            )
            feat['target'] = int(raw['target'])
            feat['key'] = value
            feat['with_format_raw'] = raw['with_format']
            rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label=""):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"label": label, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": p, "Recall": r, "F1": f}


def _format_table(rows: list[dict]) -> str:
    headers = ["Метод", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]
    col_w = [15, 6, 6, 6, 6, 11, 9, 9]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt_h = "|" + "|".join(f" {{:<{w-1}s}}" for w in col_w) + "|"
    fmt_d = "|" + "|".join([
        f" {{:<{col_w[0]-1}s}}",
        f" {{:>{col_w[1]-1}d}}",
        f" {{:>{col_w[2]-1}d}}",
        f" {{:>{col_w[3]-1}d}}",
        f" {{:>{col_w[4]-1}d}}",
        f" {{:>{col_w[5]-1}.4f}}",
        f" {{:>{col_w[6]-1}.4f}}",
        f" {{:>{col_w[7]-1}.4f}}",
    ]) + "|"

    lines = [sep, fmt_h.format(*headers), sep]
    for r in rows:
        lines.append(fmt_d.format(
            r["label"], r["TP"], r["FP"], r["FN"], r["TN"],
            r["Precision"], r["Recall"], r["F1"],
        ))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dataset_path: str, model_path: str | None = None,
         test_size: float = 0.2, seed: int = 42):

    print(f"Loading dataset from {dataset_path} …")
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    need = [d for d in raw if d.get("need")]
    print(f"Files with gettext: {len(need)} / {len(raw)}")

    print("Parsing files and computing features …")
    df = _build_dataframe(need)
    print(f"Total samples: {len(df):,}  (positive={df['target'].sum():,},"
          f" negative={(df['target'] == 0).sum():,})")

    X = df.drop(columns=["key", "target", "with_format_raw"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
    )
    test_idx = y_test.index

    keys_test = df.loc[test_idx, "key"].tolist()
    wf_test = df.loc[test_idx, "with_format_raw"].tolist()
    y_true = y_test.values

    print(f"\nTest split: {len(y_true):,} samples"
          f"  (positive={y_true.sum():,}, negative={(y_true == 0).sum():,})")

    # ------- Heuristic baseline -------
    print("\nRunning heuristic baseline …")
    h_labels = [classify_heuristic(k, wf) for k, wf in zip(keys_test, wf_test)]

    h_gray_count = h_labels.count("GRAY")
    h_in_count = h_labels.count("IN")
    h_out_count = h_labels.count("OUT")
    print(f"  Heuristic distribution: IN={h_in_count}, OUT={h_out_count}, GRAY={h_gray_count}")

    # GRAY → OUT (strict mode used for both methods)
    h_pred = [1 if l == LABEL_IN else 0 for l in h_labels]

    # ------- ML model -------
    print("Running ML model …")
    model: CatBoostClassifier = load_model(model_path)
    features_list = [
        {name: X_test.loc[idx, name] for name in FEATURE_NAMES}
        for idx in test_idx
    ]
    ml_labels = predict(model, features_list,
                        threshold_in=DEFAULT_THRESHOLD_IN,
                        threshold_out=DEFAULT_THRESHOLD_OUT)

    ml_gray_count = ml_labels.count("GRAY")
    ml_in_count = ml_labels.count("IN")
    ml_out_count = ml_labels.count("OUT")
    print(f"  ML distribution:        IN={ml_in_count}, OUT={ml_out_count}, GRAY={ml_gray_count}")

    ml_pred = [1 if l == LABEL_IN else 0 for l in ml_labels]

    # ------- Results -------
    print("\n" + "=" * 65)
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 65)

    rows = [
        _metrics(y_true, h_pred,  "Эвристика"),
        _metrics(y_true, ml_pred, "ML CatBoost"),
    ]
    table_str = _format_table(rows)
    print(table_str)

    print()
    print(f"  Серая зона: Эвристика={h_gray_count} ({100*h_gray_count/len(h_labels):.1f}%)"
          f" | ML={ml_gray_count} ({100*ml_gray_count/len(ml_labels):.1f}%)")

    # Save table to .hidden/
    import os
    hidden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hidden")
    os.makedirs(hidden_dir, exist_ok=True)
    out_path = os.path.join(hidden_dir, "baseline_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ\n")
        f.write(f"Тестовая выборка: {len(y_true):,} строк"
                f" (положительных={int(y_true.sum()):,},"
                f" отрицательных={int((y_true==0).sum()):,})\n\n")
        f.write(table_str + "\n\n")
        f.write(f"Серая зона: Эвристика={h_gray_count} ({100*h_gray_count/len(h_labels):.1f}%)"
                f" | ML={ml_gray_count} ({100*ml_gray_count/len(ml_labels):.1f}%)\n")
    print(f"\nТаблица сохранена: {out_path}")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate heuristic baseline vs ML model")
    p.add_argument("--dataset", default="dataset.json")
    p.add_argument("--model", default=None, help="Path to .cbm model file (optional)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset_path=args.dataset,
        model_path=args.model,
        test_size=args.test_size,
        seed=args.seed,
    )

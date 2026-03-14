"""
Train the CatBoost model for gettextify.

Usage:
    python train.py --dataset dataset.json [--output gettextify/model/catboost_model.cbm]
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from gettextify.features import FEATURE_NAMES, compute_features


# ---------------------------------------------------------------------------
#  Python 2 -> 3 source compatibility helper
# ---------------------------------------------------------------------------

def _convert_py2_to_py3(code: str) -> str:
    pattern_print_empty = re.compile(r'(?m)^([ \t]*)print\s*$')
    code = pattern_print_empty.sub(r'\1print()', code)

    pattern_print = re.compile(r'(?m)^([ \t]*)print\s+(?!\()(.+)$')
    code = pattern_print.sub(lambda m: f'{m.group(1)}print({m.group(2).strip()})', code)

    pattern_except = re.compile(
        r'(?m)^([ \t]*)except\s+([a-zA-Z_][\w\.]*)(\s*,\s*([a-zA-Z_][\w]*))\s*:'
    )
    code = pattern_except.sub(
        lambda m: f'{m.group(1)}except {m.group(2)} as {m.group(4)}:', code
    )

    code = re.sub(r'(?<![\w.])0([0-7]+)\b', r'0o\1', code)
    code = re.sub(r'(?<![\w.])0([1-9]\d*)\b', r'\1', code)

    return code


# ---------------------------------------------------------------------------
#  GettextFinder — AST visitor that collects labelled string data
# ---------------------------------------------------------------------------

class _GettextError(Exception):
    pass


class GettextFinder(ast.NodeVisitor):
    DEFAULT_GETTEXT = 'gettext'
    DEFAULT_GETTEXT_FUNC = [DEFAULT_GETTEXT, 'gettext_lazy']
    GETTEXT_MODULE_FUNC = (
        'dgettext', 'dngettext', 'gettext', 'lgettext', 'ldgettext',
        'ldngettext', 'lngettext', 'ngettext',
        'pgettext', 'dpgettext', 'npgettext', 'dnpgettext', 'ugettext',
    )
    GETTEXT_FUNC_ARG_NUMBER = {'dgettext': 1}
    GETTEXT_LANGUAGE_FUNC = 'language'

    def __init__(self):
        self.func_names: set[str] = set()
        self.module_aliases: set[str] = set()
        self.dataset: dict[str, dict] = {}
        self.formats: list = []
        self.mods: list = []
        self.language_vars: set[str] = set()

    # -- helpers --

    def _check_gettext(self, word: str) -> bool:
        if word in self.DEFAULT_GETTEXT_FUNC:
            return True
        return self.DEFAULT_GETTEXT in word

    def _is_gettext_name_func(self, node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.func_names
        )

    def _is_gettext_attr_func(self, node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.module_aliases
            and node.func.attr in self.GETTEXT_MODULE_FUNC
        )

    def _is_gettext_language_call(self, node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.module_aliases
        )

    def _append_dataset(self, node, *, target: bool = False):
        with_inner_format = False

        if isinstance(node, ast.Constant):
            value = node.value
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
        ):
            value = node.left.value
            with_inner_format = True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Constant)
            and node.func.attr == 'format'
        ):
            value = node.func.value.value
            with_inner_format = True
        elif (
            isinstance(node, ast.JoinedStr)
            and len(node.values) > 1
            and isinstance(node.values[0], ast.Constant)
        ):
            value = node.values[0].value
            with_inner_format = True
        else:
            raise _GettextError(f'{node}, {node.__dict__}')

        if value not in self.dataset:
            self.dataset[value] = {
                'value': value,
                'count': 0,
                'with_format': False,
                'with_inner_format': with_inner_format,
                'target': target,
            }
        self.dataset[value]['count'] += 1
        if target and not self.dataset[value]['target']:
            self.dataset[value]['target'] = target

    # -- format / mod post-processing --

    def _process_format(self, node):
        base_node = node.func.value
        if self._is_gettext_name_func(base_node) and len(base_node.args) > 0:
            if not isinstance(base_node.args[0], ast.Constant):
                raise _GettextError()
            value = base_node.args[0].value
            self.dataset[value]['with_format'] = True
        elif self._is_gettext_attr_func(base_node):
            raise _GettextError()

    def _process_mod(self, node):
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Call)
            and self._is_gettext_name_func(node.left)
            and len(node.left.args) > 0
        ):
            if not isinstance(node.left.args[0], ast.Constant):
                raise _GettextError(
                    f'Gettext func but left is not constant, {node}, {node.__dict__}'
                )
            value = node.left.args[0].value
            self.dataset[value]['with_format'] = True

    # -- visitors --

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
        if isinstance(node.func, ast.Name):
            self._visit_call_name(node)
        elif isinstance(node.func, ast.Attribute):
            self._visit_call_attribute(node)
        self.generic_visit(node)

    def _visit_call_name(self, node):
        if self._is_gettext_name_func(node) and len(node.args) > 0:
            arg_no = self.GETTEXT_FUNC_ARG_NUMBER.get(node.func.id, 0)
            self._append_dataset(node.args[arg_no], target=True)

    def _visit_call_attribute(self, node):
        if node.func.attr == 'format':
            self.formats.append(node)
            return
        if self._is_gettext_attr_func(node) and len(node.args) > 0:
            self._append_dataset(node.args[0], target=True)

    def visit_Assign(self, node):
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr in self.GETTEXT_MODULE_FUNC
            and (
                self._is_gettext_language_call(node.value.value)
                or (
                    isinstance(node.value.value, ast.Name)
                    and node.value.value.id in self.language_vars
                )
            )
        ):
            if isinstance(node.targets[0], ast.Name):
                self.func_names.add(node.targets[0].id)
        elif self._is_gettext_language_call(node.value):
            if isinstance(node.targets[0], ast.Name):
                self.language_vars.add(node.targets[0].id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mod):
            self.mods.append(node)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self._append_dataset(node, target=False)
        self.generic_visit(node)

    # -- entry point --

    def run(self, tree):
        self.visit(tree)
        for node in self.formats:
            self._process_format(node)
        for node in self.mods:
            self._process_mod(node)


# ---------------------------------------------------------------------------
#  Feature computation for dataset items
# ---------------------------------------------------------------------------

def _build_features_row(raw: dict, global_count: int = 1) -> dict:
    """Build a feature dict for one string literal using gettextify.features."""
    return compute_features(
        key=raw['value'],
        with_format=raw['with_format'],
        count=raw['count'],
        global_count=global_count,
    )


# ---------------------------------------------------------------------------
#  Dataset construction from raw GitHub JSON
# ---------------------------------------------------------------------------

def _build_dataset(need_dataset: list[dict]) -> pd.DataFrame:
    """
    Iterate over collected GitHub Python files, parse each with
    GettextFinder, compute features, and return a DataFrame.
    """
    all_rows: list[dict] = []
    global_index: dict[str, int] = {}

    for i, item in enumerate(need_dataset):
        source = item['file']

        try:
            tree = ast.parse(source)
        except SyntaxError:
            source = _convert_py2_to_py3(source)
            try:
                tree = ast.parse(source)
            except SyntaxError as err:
                print(f'[{i}] SyntaxError, skipping: {err}', file=sys.stderr)
                continue

        finder = GettextFinder()
        try:
            finder.run(tree)
        except _GettextError as err:
            print(f'[{i}] GettextError, skipping: {err}', file=sys.stderr)
            continue

        for value, raw in finder.dataset.items():
            if value in global_index:
                global_index[value] += 1
            else:
                global_index[value] = 1

            row = _build_features_row(raw, global_count=global_index[value])
            row['target'] = int(raw['target'])
            row['key'] = value
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    output_path: str,
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.1,
    depth: int = 6,
    verbose_every: int = 50,
):
    print(f'Loading dataset from {dataset_path} …')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)

    need_dataset = [d for d in raw_dataset if d.get('need')]
    print(f'Files with gettext import: {len(need_dataset)} / {len(raw_dataset)}')

    print('Parsing files and computing features …')
    df = _build_dataset(need_dataset)
    print(f'Total samples: {len(df)}')
    print(f'Target distribution:\n{df["target"].value_counts().to_string()}')

    X = df.drop(columns=['key', 'target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed,
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    cat_features = list(X_train.columns)

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        class_weights=class_weights,
        eval_metric='F1',
        random_seed=random_seed,
        verbose=verbose_every,
    )

    print('\nTraining CatBoost model …')
    model.fit(train_pool, eval_set=test_pool)

    # -- evaluation --
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('\n--- Evaluation on test set ---')
    print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_test, y_pred):.4f}')
    print(f'F1:        {f1_score(y_test, y_pred):.4f}')
    print(f'ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}')
    print()
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred))

    # -- optimal threshold search --
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [
        f1_score(y_test, (y_proba >= th).astype(int)) for th in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = f1_scores[best_idx]
    print(f'Best F1 = {best_f1:.4f}  at threshold = {best_threshold:.2f}')

    # -- feature importance --
    feat_imp = model.get_feature_importance()
    print('\n--- Feature importance ---')
    for name, imp in sorted(zip(cat_features, feat_imp), key=lambda x: -x[1]):
        print(f'  {name:25s} {imp:.4f}')

    # -- save --
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    print(f'\nModel saved to {out}')
    print(
        f'Suggested DEFAULT_THRESHOLD for predictor.py: {best_threshold:.2f}'
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train the CatBoost model for gettextify',
    )
    parser.add_argument(
        '--dataset', required=True,
        help='Path to dataset.json (list of {url, file, need})',
    )
    parser.add_argument(
        '--output', default='gettextify/model/catboost_model.cbm',
        help='Output model path (default: gettextify/model/catboost_model.cbm)',
    )
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--depth', type=int, default=6)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_path=args.output,
        test_size=args.test_size,
        random_seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
    )


if __name__ == '__main__':
    main()

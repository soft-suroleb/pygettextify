from __future__ import annotations

import os

from catboost import CatBoostClassifier, Pool

from .features import FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")

# Default thresholds: P(IN) >= threshold_in → IN
#                     P(IN) <  1 - threshold_out → OUT
#                     otherwise → GRAY
DEFAULT_THRESHOLD_IN = 0.73
DEFAULT_THRESHOLD_OUT = 0.73

# Backward-compat alias
DEFAULT_THRESHOLD = DEFAULT_THRESHOLD_IN

LABEL_IN = "IN"
LABEL_OUT = "OUT"
LABEL_GRAY = "GRAY"


def load_model(path: str | None = None) -> CatBoostClassifier:
    if path and os.path.isfile(path):
        resolved = path
    elif path:
        print(f"Specified model not found: {path}, using default model")
        resolved = MODEL_PATH
    else:
        resolved = MODEL_PATH

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Model not found: {resolved}\n"
            "Train the model first: python -m gettext_tool.train"
        )
    model = CatBoostClassifier()
    model.load_model(resolved)
    return model


def predict(
    model: CatBoostClassifier,
    features_list: list[dict],
    threshold: float | None = None,
    threshold_in: float | None = None,
    threshold_out: float | None = None,
) -> list[str]:
    """Classify each string as LABEL_IN, LABEL_OUT, or LABEL_GRAY.

    Args:
        model: loaded CatBoost model.
        features_list: list of feature dicts from compute_features().
        threshold: convenience shorthand — sets both threshold_in and
            threshold_out to this value (ignored when threshold_in or
            threshold_out are given explicitly).
        threshold_in: P(IN) >= threshold_in  → LABEL_IN.
        threshold_out: P(IN) <  1-threshold_out → LABEL_OUT, else LABEL_GRAY.
    """
    if not features_list:
        return []

    t_in = threshold_in if threshold_in is not None else (
        threshold if threshold is not None else DEFAULT_THRESHOLD_IN
    )
    t_out = threshold_out if threshold_out is not None else (
        threshold if threshold is not None else DEFAULT_THRESHOLD_OUT
    )

    data = [[f[name] for name in FEATURE_NAMES] for f in features_list]
    pool = Pool(data=data, feature_names=FEATURE_NAMES, cat_features=FEATURE_NAMES)
    probas = model.predict_proba(pool)[:, 1]

    labels = []
    for p in probas:
        p = float(p)
        if p >= t_in:
            labels.append(LABEL_IN)
        elif p < 1.0 - t_out:
            labels.append(LABEL_OUT)
        else:
            labels.append(LABEL_GRAY)
    return labels

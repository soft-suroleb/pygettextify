import os

from catboost import CatBoostClassifier, Pool

from .features import FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")
DEFAULT_THRESHOLD = 0.78


def load_model(path: str | None = None) -> CatBoostClassifier:
    if path and os.path.isfile(path):
        resolved = path
    elif path:
        print(f"Указанная модель не найдена: {path}, используется модель по умолчанию")
        resolved = MODEL_PATH
    else:
        resolved = MODEL_PATH

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Модель не найдена: {resolved}\n"
            "Сначала обучите модель: python -m gettext_tool.train"
        )
    model = CatBoostClassifier()
    model.load_model(resolved)
    return model


def predict(
    model: CatBoostClassifier,
    features_list: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[bool]:
    if not features_list:
        return []
    data = [[f[name] for name in FEATURE_NAMES] for f in features_list]
    pool = Pool(data=data, feature_names=FEATURE_NAMES, cat_features=FEATURE_NAMES)
    probas = model.predict_proba(pool)[:, 1]
    return [float(p) >= threshold for p in probas]

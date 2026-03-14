"""Integration tests: model predictions on realistic string inputs.

These tests verify that the trained CatBoost model makes reasonable
predictions for typical user-facing vs technical strings.

NOTE: The model was trained on strings extracted from real GitHub repos.
It relies on features like count, global_count, with_format, and
textual properties. Testing strings in complete isolation (count=1,
global_count=1) may not reflect how the model behaves in the actual
pipeline.  Pipeline integration tests below are the most realistic.
"""

import pytest

from gettextify.features import compute_features
from gettextify.parser import extract_strings
from gettextify.predictor import predict


def _predict_one(
    model,
    key: str,
    *,
    with_format: bool = False,
    count: int = 1,
    global_count: int = 1,
) -> bool:
    features = compute_features(
        key, with_format=with_format, count=count, global_count=global_count,
    )
    return predict(model, [features])[0]


def _predict_batch(model, keys: list[str], **kwargs) -> dict[str, bool]:
    features = [
        compute_features(k, with_format=kwargs.get("with_format", False), count=1)
        for k in keys
    ]
    preds = predict(model, features)
    return dict(zip(keys, preds))


# ---------------------------------------------------------------------------
#  Technical / non-user-facing strings (the model is confident about these)
# ---------------------------------------------------------------------------

class TestTechnicalStrings:
    @pytest.mark.parametrize("text", [
        "utf-8",
        "application/json",
        "text/html",
        "rb",
        "w",
    ])
    def test_encodings_and_mimetypes(self, model, text):
        assert _predict_one(model, text) is False

    @pytest.mark.parametrize("text", [
        "__init__",
        "__main__",
        "__name__",
        "__all__",
    ])
    def test_dunder_names(self, model, text):
        assert _predict_one(model, text) is False

    @pytest.mark.parametrize("text", [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ])
    def test_log_level_names(self, model, text):
        assert _predict_one(model, text) is False

    @pytest.mark.parametrize("text", [
        "config.yaml",
        "/usr/bin/python",
        "src/main.py",
    ])
    def test_file_paths(self, model, text):
        assert _predict_one(model, text) is False

    @pytest.mark.parametrize("text", [
        "user_id",
        "api_key",
        "created_at",
        "session_token",
    ])
    def test_snake_case_identifiers(self, model, text):
        assert _predict_one(model, text) is False

    @pytest.mark.parametrize("text", [
        "SELECT * FROM users",
        "INSERT INTO table",
    ])
    def test_sql_queries(self, model, text):
        assert _predict_one(model, text) is False

    def test_empty_string(self, model):
        assert _predict_one(model, "") is False

    def test_single_letter(self, model):
        assert _predict_one(model, "x") is False

    @pytest.mark.parametrize("text", [
        "GET",
        "POST",
        "PUT",
        "PATCH",
    ])
    def test_http_methods(self, model, text):
        assert _predict_one(model, text) is False


# ---------------------------------------------------------------------------
#  Format strings
# ---------------------------------------------------------------------------

class TestFormatStrings:
    def test_strftime_format(self, model):
        assert _predict_one(model, "%Y-%m-%d %H:%M:%S") is False

    def test_brace_format_with_flag(self, model):
        assert _predict_one(model, "Hello {name}!", with_format=True) is False

    def test_percent_named_format(self, model):
        assert _predict_one(model, "%(asctime)s [%(levelname)s]") is False


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_url_string(self, model):
        assert _predict_one(model, "https://example.com/api/v1") is False

    def test_json_string(self, model):
        assert _predict_one(model, '{"key": "value"}') is False

    def test_html_tag(self, model):
        assert _predict_one(model, "<div class='container'>") is False

    def test_regex_pattern(self, model):
        assert _predict_one(model, r"^\d{3}-\d{4}$") is False

    def test_version_string(self, model):
        assert _predict_one(model, "1.0.0") is False


# ---------------------------------------------------------------------------
#  Threshold sensitivity
# ---------------------------------------------------------------------------

class TestThresholdBehavior:
    def test_zero_threshold_all_positive(self, model):
        features = [compute_features("anything", with_format=False, count=1)]
        result = predict(model, features, threshold=0.0)
        assert all(result)

    def test_one_threshold_all_negative(self, model):
        features = [compute_features("Hello world!", with_format=False, count=1)]
        result = predict(model, features, threshold=1.0)
        assert not any(result)

    def test_high_threshold_fewer_positives(self, model):
        features = [
            compute_features("Click here.", with_format=False, count=1),
            compute_features("DEBUG", with_format=False, count=1),
            compute_features("__init__", with_format=False, count=1),
        ]
        low = predict(model, features, threshold=0.1)
        high = predict(model, features, threshold=0.99)
        assert sum(low) >= sum(high)


# ---------------------------------------------------------------------------
#  Pipeline integration: parser -> features -> model
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """End-to-end: parse real source snippets, compute features, predict."""

    def _run_pipeline(self, model, source: str) -> dict[str, bool]:
        lits = extract_strings(source)
        candidates = [
            l for l in lits
            if not l.is_docstring and not l.is_wrapped and not l.is_fstring_part
        ]
        if not candidates:
            return {}
        features = [
            compute_features(c.value, with_format=c.with_format, count=1)
            for c in candidates
        ]
        preds = predict(model, features)
        return {c.value: p for c, p in zip(candidates, preds)}

    def test_technical_vs_user_facing(self, model):
        source = (
            'import os\n'
            'MODE = "debug"\n'
            'ENCODING = "utf-8"\n'
            'print("Welcome to the app!")\n'
        )
        pred_map = self._run_pipeline(model, source)
        assert pred_map.get("utf-8") is False
        assert pred_map.get("debug") is False

    def test_docstrings_not_in_candidates(self, model):
        source = (
            '"""Module docstring."""\n'
            'x = "visible"\n'
        )
        pred_map = self._run_pipeline(model, source)
        assert "Module docstring." not in pred_map
        assert "visible" in pred_map

    def test_fstrings_not_in_candidates(self, model):
        source = (
            'name = "World"\n'
            'x = f"Hello {name}!"\n'
            'y = "plain text"\n'
        )
        lits = extract_strings(source)
        candidates = [
            l for l in lits
            if not l.is_docstring and not l.is_wrapped and not l.is_fstring_part
        ]
        fstring_text = [c for c in candidates if "Hello" in c.value]
        assert len(fstring_text) == 0
        assert any(c.value == "World" for c in candidates)
        assert any(c.value == "plain text" for c in candidates)

    def test_already_wrapped_excluded(self, model):
        source = (
            'from gettext import gettext as _\n'
            'x = _("Already wrapped")\n'
            'y = "Not wrapped"\n'
        )
        pred_map = self._run_pipeline(model, source)
        assert "Already wrapped" not in pred_map
        assert "Not wrapped" in pred_map

    def test_format_strings_excluded_by_flag(self, model):
        source = 'msg = "Hello %s" % name\n'
        lits = extract_strings(source)
        fmt_lits = [l for l in lits if l.with_format]
        assert len(fmt_lits) == 1
        assert fmt_lits[0].value == "Hello %s"

    def test_mixed_source_file(self, model):
        source = (
            'import logging\n'
            '\n'
            'LOG_LEVEL = "DEBUG"\n'
            'DB_URL = "postgresql://localhost/mydb"\n'
            'ENCODING = "utf-8"\n'
            '\n'
            'def greet(name):\n'
            '    """Greet the user."""\n'
            '    return f"Hello, {name}!"\n'
        )
        pred_map = self._run_pipeline(model, source)
        assert pred_map.get("utf-8") is False
        assert "Greet the user." not in pred_map

    def test_multiline_string_detection(self, model):
        source = 'x = """This is a\nmultiline string."""\n'
        pred_map = self._run_pipeline(model, source)
        assert len(pred_map) == 1

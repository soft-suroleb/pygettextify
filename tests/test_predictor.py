"""Tests for gettextify.predictor — model loading and prediction."""

import os

import pytest

from gettextify.features import FEATURE_NAMES, compute_features
from gettextify.predictor import (
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLD_IN,
    DEFAULT_THRESHOLD_OUT,
    LABEL_GRAY,
    LABEL_IN,
    LABEL_OUT,
    MODEL_PATH,
    load_model,
    predict,
)


class TestLoadModel:
    def test_loads_default_model(self, model):
        assert model is not None

    def test_model_file_exists(self):
        assert os.path.isfile(MODEL_PATH)

    def test_fallback_to_default_on_bad_path(self):
        """When a non-existent path is given, load_model falls back to the default."""
        model = load_model("/totally/fake/path/model.cbm")
        assert model is not None

    def test_raises_when_no_model_at_all(self, tmp_path, monkeypatch):
        """If both given path and default path are missing, FileNotFoundError is raised."""
        monkeypatch.setattr(
            "gettextify.predictor.MODEL_PATH",
            str(tmp_path / "missing_default.cbm"),
        )
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "also_missing.cbm"))


class TestPredict:
    def test_empty_features_returns_empty(self, model):
        result = predict(model, [])
        assert result == []

    def test_returns_list_of_strings(self, model):
        features = [compute_features("hello world", with_format=False, count=1)]
        result = predict(model, features)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_labels_are_valid(self, model):
        features = [compute_features("hello world", with_format=False, count=1)]
        result = predict(model, features)
        valid = {LABEL_IN, LABEL_OUT, LABEL_GRAY}
        assert all(r in valid for r in result)

    def test_returns_correct_length(self, model):
        features = [
            compute_features("one", with_format=False, count=1),
            compute_features("two", with_format=False, count=1),
            compute_features("three", with_format=False, count=1),
        ]
        result = predict(model, features)
        assert len(result) == 3

    def test_high_threshold_in_fewer_positives(self, model):
        features = [
            compute_features("Click here to continue.", with_format=False, count=1),
            compute_features("DEBUG", with_format=False, count=1),
            compute_features("__init__", with_format=False, count=1),
        ]
        result_low = predict(model, features, threshold_in=0.1, threshold_out=0.1)
        result_high = predict(model, features, threshold_in=0.99, threshold_out=0.99)
        in_low = sum(1 for r in result_low if r == LABEL_IN)
        in_high = sum(1 for r in result_high if r == LABEL_IN)
        assert in_low >= in_high

    def test_threshold_shorthand(self, model):
        features = [compute_features("hello world", with_format=False, count=1)]
        result_via_shorthand = predict(model, features, threshold=0.5)
        result_via_explicit = predict(model, features, threshold_in=0.5, threshold_out=0.5)
        assert result_via_shorthand == result_via_explicit

    def test_zero_threshold_in_all_in_or_gray(self, model):
        """With threshold_in=0 everything is IN (no OUT possible if threshold_out=1)."""
        features = [compute_features("anything", with_format=False, count=1)]
        result = predict(model, features, threshold_in=0.0, threshold_out=1.0)
        assert all(r == LABEL_IN for r in result)

    def test_extreme_thresholds_produce_all_gray(self, model):
        """With threshold_in=1 and threshold_out=1 everything falls into GRAY."""
        features = [
            compute_features("Click here to continue.", with_format=False, count=1),
            compute_features("DEBUG", with_format=False, count=1),
        ]
        result = predict(model, features, threshold_in=1.0, threshold_out=1.0)
        assert all(r == LABEL_GRAY for r in result)

    def test_obvious_in_string(self, model):
        # With threshold_in=0.0 and threshold_out=1.0, P(IN) >= 0 is always true,
        # so every string must be classified as IN regardless of the model score.
        features = [compute_features("Click here to continue.", with_format=False, count=1)]
        result = predict(model, features, threshold_in=0.0, threshold_out=1.0)
        assert result[0] == LABEL_IN

    def test_obvious_out_string(self, model):
        features = [compute_features("DEBUG", with_format=False, count=1)]
        result = predict(model, features)
        assert result[0] == LABEL_OUT


class TestDefaultThreshold:
    def test_threshold_in_valid_range(self):
        assert 0.0 < DEFAULT_THRESHOLD < 1.0
        assert 0.0 < DEFAULT_THRESHOLD_IN < 1.0
        assert 0.0 < DEFAULT_THRESHOLD_OUT < 1.0

    def test_threshold_is_float(self):
        assert isinstance(DEFAULT_THRESHOLD, float)
        assert isinstance(DEFAULT_THRESHOLD_IN, float)
        assert isinstance(DEFAULT_THRESHOLD_OUT, float)

    def test_backward_compat_alias(self):
        assert DEFAULT_THRESHOLD == DEFAULT_THRESHOLD_IN

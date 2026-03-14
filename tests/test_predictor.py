"""Tests for gettextify.predictor — model loading and prediction."""

import os

import pytest

from gettextify.features import FEATURE_NAMES, compute_features
from gettextify.predictor import DEFAULT_THRESHOLD, MODEL_PATH, load_model, predict


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

    def test_returns_list_of_bools(self, model):
        features = [compute_features("hello world", with_format=False, count=1)]
        result = predict(model, features)
        assert isinstance(result, list)
        assert all(isinstance(r, bool) for r in result)

    def test_returns_correct_length(self, model):
        features = [
            compute_features("one", with_format=False, count=1),
            compute_features("two", with_format=False, count=1),
            compute_features("three", with_format=False, count=1),
        ]
        result = predict(model, features)
        assert len(result) == 3

    def test_high_threshold_fewer_positives(self, model):
        features = [
            compute_features("Click here to continue.", with_format=False, count=1),
            compute_features("DEBUG", with_format=False, count=1),
            compute_features("__init__", with_format=False, count=1),
        ]
        result_low = predict(model, features, threshold=0.1)
        result_high = predict(model, features, threshold=0.99)
        assert sum(result_low) >= sum(result_high)

    def test_zero_threshold_all_positive(self, model):
        features = [
            compute_features("anything", with_format=False, count=1),
        ]
        result = predict(model, features, threshold=0.0)
        assert all(result)


class TestDefaultThreshold:
    def test_threshold_in_valid_range(self):
        assert 0.0 < DEFAULT_THRESHOLD < 1.0

    def test_threshold_is_float(self):
        assert isinstance(DEFAULT_THRESHOLD, float)

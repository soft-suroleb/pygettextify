import pytest

from gettextify.predictor import load_model, MODEL_PATH


@pytest.fixture(scope="session")
def model():
    """Load the CatBoost model once for the entire test session."""
    return load_model(MODEL_PATH)

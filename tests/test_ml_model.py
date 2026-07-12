"""
tests/test_ml_model.py
Tests for ml_model.py - verifies training produces working models
and predictions come back in sensible ranges.
"""

import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))

import ml_model


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory, monkeypatch_module=None):
    """
    Trains models once using a small synthetic dataset, saved to a
    temporary directory - never overwrites your real trained models
    in the models/ folder.

    scope="module" means this fixture runs ONCE for all tests in this
    file, not once per test - training is slow enough that re-running
    it for every single test would waste time unnecessarily.
    """
    tmp_dir = tmp_path_factory.mktemp("models")

    original_recovery_path = ml_model.RECOVERY_MODEL_PATH
    original_grade_path = ml_model.GRADE_MODEL_PATH

    ml_model.RECOVERY_MODEL_PATH = tmp_dir / "recovery_model.pkl"
    ml_model.GRADE_MODEL_PATH = tmp_dir / "grade_model.pkl"

    df = ml_model.generate_synthetic_training_data(n_samples=200, random_seed=1)
    ml_model.train_models(df)

    yield

    # Restore original paths so we don't affect other tests/files
    ml_model.RECOVERY_MODEL_PATH = original_recovery_path
    ml_model.GRADE_MODEL_PATH = original_grade_path


def test_generate_synthetic_training_data_has_expected_columns():
    """
    Confirms our synthetic data generator produces all required columns
    with no missing values.
    """
    df = ml_model.generate_synthetic_training_data(n_samples=50)

    expected_columns = {"avg_bubble_size", "color_hue_avg", "texture_score", "froth_speed", "recovery", "grade"}
    assert expected_columns.issubset(set(df.columns))
    assert df.isnull().sum().sum() == 0  # no missing/NaN values anywhere


def test_synthetic_recovery_within_realistic_bounds():
    """
    Confirms our synthetic recovery values stay within the realistic
    clipped range we defined (40-98%), even with random noise added.
    """
    df = ml_model.generate_synthetic_training_data(n_samples=200)

    assert df["recovery"].min() >= 40
    assert df["recovery"].max() <= 98


def test_predict_recovery_grade_returns_valid_ranges(trained_models):
    """
    Runs a prediction on a made-up (but realistic) feature set and
    confirms the output is a sensible percentage, not something like
    a negative number or 500%.
    """
    fake_features = {
        "avg_bubble_size": 400.0,
        "color_hue_avg": 65.0,
        "texture_score": 300.0,
        "froth_speed": 1.5,
    }

    result = ml_model.predict_recovery_grade(fake_features)

    assert 0 <= result["predicted_recovery"] <= 100
    assert 0 <= result["predicted_grade"] <= 100


def test_predict_recovery_grade_returns_expected_keys(trained_models):
    """
    Confirms the prediction function returns exactly the keys the
    rest of our app (app.py) depends on.
    """
    fake_features = {
        "avg_bubble_size": 400.0,
        "color_hue_avg": 65.0,
        "texture_score": 300.0,
        "froth_speed": 1.5,
    }

    result = ml_model.predict_recovery_grade(fake_features)

    assert "predicted_recovery" in result
    assert "predicted_grade" in result
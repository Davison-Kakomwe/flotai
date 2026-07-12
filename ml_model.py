"""
ml_model.py
Trains and applies a machine learning model that predicts copper recovery %
and concentrate grade % from froth features extracted by cv_features.py.

IMPORTANT (hackathon honesty note):
We do not have real lab assay data for this prototype, so training labels
are SYNTHETIC - generated from a formula approximating known relationships
between froth appearance and flotation performance. In production, this
would be replaced by real lab assay data via the lab_assays table already
present in our database schema (see database.py).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

RECOVERY_MODEL_PATH = MODEL_DIR / "recovery_model.pkl"
GRADE_MODEL_PATH = MODEL_DIR / "grade_model.pkl"


def generate_synthetic_training_data(n_samples=500, random_seed=42):
    """
    Generates a synthetic dataset of froth features + corresponding
    recovery/grade labels, using a formula loosely based on known
    flotation relationships, plus random noise (real-world data is
    never perfectly clean).

    Ranges below are chosen to roughly match what we observed from
    our real test videos in cv_features.py.
    """
    rng = np.random.default_rng(random_seed)

    avg_bubble_size = rng.uniform(50, 1800, n_samples)
    color_hue_avg = rng.uniform(10, 140, n_samples)
    texture_score = rng.uniform(50, 1800, n_samples)
    froth_speed = rng.uniform(0.0, 4.0, n_samples)

    # Synthetic formula: smaller bubbles + moderate hue + lower texture
    # variance + moderate speed -> higher recovery (a plausible pattern,
    # not derived from real plant data)
    recovery = (
        88
        - 0.015 * avg_bubble_size
        - 0.02 * np.abs(color_hue_avg - 60)
        - 0.01 * texture_score
        + 2.0 * froth_speed
        + rng.normal(0, 3, n_samples)  # random noise, like real-world variability
    )
    recovery = np.clip(recovery, 40, 98)  # keep within realistic bounds

    grade = (
        26
        - 0.004 * avg_bubble_size
        - 0.01 * np.abs(color_hue_avg - 60)
        - 0.003 * texture_score
        + 0.5 * froth_speed
        + rng.normal(0, 1.5, n_samples)
    )
    grade = np.clip(grade, 10, 35)

    df = pd.DataFrame({
        "avg_bubble_size": avg_bubble_size,
        "color_hue_avg": color_hue_avg,
        "texture_score": texture_score,
        "froth_speed": froth_speed,
        "recovery": recovery,
        "grade": grade,
    })

    return df


def train_models(df):
    """
    Trains two separate RandomForestRegressor models: one for recovery,
    one for grade. We use RandomForest because:
    - It handles non-linear relationships without needing feature scaling
    - It's robust to noisy data (important since our labels have noise)
    - It's easy to explain to non-technical judges: "many small decision
      trees vote and we average their answers"
    """
    feature_cols = ["avg_bubble_size", "color_hue_avg", "texture_score", "froth_speed"]
    X = df[feature_cols]

    y_recovery = df["recovery"]
    y_grade = df["grade"]

    # Split into training and testing sets - we train on 80%, test on
    # the remaining 20% to check how well the model generalizes to
    # data it hasn't seen
    X_train, X_test, y_rec_train, y_rec_test = train_test_split(
        X, y_recovery, test_size=0.2, random_state=42
    )
    _, _, y_grade_train, y_grade_test = train_test_split(
        X, y_grade, test_size=0.2, random_state=42
    )

    recovery_model = RandomForestRegressor(n_estimators=200, random_state=42)
    recovery_model.fit(X_train, y_rec_train)

    grade_model = RandomForestRegressor(n_estimators=200, random_state=42)
    grade_model.fit(X_train, y_grade_train)

    # Evaluate on the held-out test set
    rec_predictions = recovery_model.predict(X_test)
    grade_predictions = grade_model.predict(X_test)

    print("Recovery model performance (on held-out test data):")
    print(f"  MAE: {mean_absolute_error(y_rec_test, rec_predictions):.2f}")
    print(f"  R2 score: {r2_score(y_rec_test, rec_predictions):.3f}")

    print("Grade model performance (on held-out test data):")
    print(f"  MAE: {mean_absolute_error(y_grade_test, grade_predictions):.2f}")
    print(f"  R2 score: {r2_score(y_grade_test, grade_predictions):.3f}")

    # Save both trained models to disk so we don't need to retrain
    # every time the app runs
    joblib.dump(recovery_model, RECOVERY_MODEL_PATH)
    joblib.dump(grade_model, GRADE_MODEL_PATH)
    print(f"\nModels saved to {MODEL_DIR}")

    return recovery_model, grade_model


def load_models():
    """
    Loads previously trained models from disk.
    Called by app.py at runtime - we never want to retrain the model
    every time someone opens the dashboard, only when we explicitly
    want to update it.
    """
    recovery_model = joblib.load(RECOVERY_MODEL_PATH)
    grade_model = joblib.load(GRADE_MODEL_PATH)
    return recovery_model, grade_model


def predict_recovery_grade(features: dict):
    """
    Takes a dictionary of froth features (as returned by
    cv_features.extract_all_features) and returns predicted
    recovery % and grade %.
    """
    recovery_model, grade_model = load_models()

    feature_order = ["avg_bubble_size", "color_hue_avg", "texture_score", "froth_speed"]
    X = pd.DataFrame([{
        key: features[key] for key in feature_order
    }])

    predicted_recovery = float(recovery_model.predict(X)[0])
    predicted_grade = float(grade_model.predict(X)[0])

    return {
        "predicted_recovery": round(predicted_recovery, 2),
        "predicted_grade": round(predicted_grade, 2),
    }


if __name__ == "__main__":
    print("Generating synthetic training data...\n")
    df = generate_synthetic_training_data(n_samples=500)
    print(df.head())
    print(f"\nGenerated {len(df)} synthetic samples.\n")

    print("Training models...\n")
    train_models(df)

    print("\n--- Testing prediction on real extracted video features ---\n")
    from cv_features import extract_all_features
    from pathlib import Path as _Path

    videos_folder = _Path("videos")
    video_files = list(videos_folder.glob("*.mp4"))

    if video_files:
        sample_video = video_files[0]
        real_features = extract_all_features(sample_video)
        prediction = predict_recovery_grade(real_features)
        print(f"Video: {sample_video.name}")
        print(f"Extracted features: {real_features}")
        print(f"Prediction: {prediction}")
    else:
        print("No videos found to test prediction on.")
"""
tests/test_cv_features.py
Unit tests for cv_features.py - verifies our computer vision functions
return sensible values on real image/video data.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

import cv_features


def make_fake_frame(color=(100, 150, 200)):
    """
    Creates a simple solid-color test image (100x100 pixels) instead of
    needing a real video file. This lets us test our CV functions fast
    and predictably, without depending on external files.
    """
    frame = np.full((100, 100, 3), color, dtype=np.uint8)
    return frame


def test_extract_color_feature_returns_valid_hue_range():
    """
    Hue in OpenCV's HSV representation is always between 0 and 179.
    Any function returning outside this range would indicate a bug.
    """
    frame = make_fake_frame()
    hue = cv_features.extract_color_feature(frame)

    assert 0 <= hue <= 179


def test_extract_texture_score_is_zero_for_flat_color():
    """
    A perfectly solid-color image has zero texture/edges - the
    Laplacian variance should be exactly (or very close to) 0.
    """
    frame = make_fake_frame()
    texture = cv_features.extract_texture_score(frame)

    assert texture == pytest.approx(0, abs=1e-6)


def test_extract_bubble_size_handles_blank_image_gracefully():
    """
    A blank/flat image has no distinguishable 'bubbles' - our function
    should return 0.0 rather than crashing.
    """
    frame = make_fake_frame()
    bubble_size = cv_features.extract_bubble_size(frame)

    assert bubble_size == 0.0


def test_extract_froth_speed_is_zero_for_identical_frames():
    """
    If two frames are identical, there's no motion between them -
    optical flow speed should be at or near 0.
    """
    frame1 = make_fake_frame()
    frame2 = make_fake_frame()

    speed = cv_features.extract_froth_speed(frame1, frame2)

    assert speed == pytest.approx(0, abs=1e-3)


def test_extract_all_features_on_real_video():
    """
    Integration test: runs the full pipeline on one real video file
    from our videos/ folder, confirming all four features come back
    as valid numbers (not None, not NaN).
    """
    videos_folder = Path(__file__).parent.parent / "videos"
    video_files = list(videos_folder.glob("*.mp4"))

    if not video_files:
        pytest.skip("No video files available for integration test")

    features = cv_features.extract_all_features(video_files[0])

    for key, value in features.items():
        assert isinstance(value, float)
        assert not np.isnan(value)
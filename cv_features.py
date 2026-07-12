"""
cv_features.py
Computer vision functions that extract measurable froth characteristics
from images and video, mirroring what a human operator visually judges.
"""

import cv2
import numpy as np


def get_frames_from_video(video_path, num_frames=2):
    """
    Reads a short video/image-sequence file and returns the first
    `num_frames` frames as a list of images (numpy arrays).
    We need at least 2 frames to calculate froth speed via optical flow.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while len(frames) < num_frames:
        success, frame = cap.read()
        if not success:
            break  # video ended before we got enough frames
        frames.append(frame)

    cap.release()  # always release video resources when done
    return frames


def extract_color_feature(frame):
    """
    Converts the frame to HSV color space and returns the average hue.
    Hue isolates 'what color' independent of brightness/lighting -
    useful because plant lighting conditions vary a lot.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_hue = float(np.mean(hsv[:, :, 0]))  # channel 0 = Hue
    return avg_hue


def extract_bubble_size(frame):
    """
    Estimates average bubble size using thresholding and contour detection.

    Steps:
    1. Convert to grayscale (bubble edges show up as intensity changes)
    2. Apply adaptive threshold to highlight bubble boundaries
    3. Find contours (outlines of connected regions)
    4. Return the average area of detected contours as a proxy for bubble size
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles uneven lighting better than a fixed threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0  # no bubbles detected - avoid crashing on empty froth images

    areas = [cv2.contourArea(c) for c in contours]
    # Filter out tiny noise specks (adjust threshold as we test on real images)
    areas = [a for a in areas if a > 5]

    if not areas:
        return 0.0

    avg_bubble_area = float(np.mean(areas))
    return avg_bubble_area


def extract_texture_score(frame):
    """
    Measures froth texture using Laplacian variance.
    Smooth, stable froth = low variance.
    Chunky, unstable, or overly aerated froth = high variance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = float(laplacian.var())
    return texture_score


def extract_froth_speed(frame1, frame2):
    """
    Calculates froth movement speed using Farneback optical flow -
    comparing pixel movement between two consecutive frames.

    Returns the average motion magnitude across the whole image,
    which acts as our 'froth speed' feature.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # flow has an (x, y) movement vector per pixel - we compute magnitude
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_speed = float(np.mean(magnitude))
    return avg_speed


def extract_all_features(video_path):
    """
    Orchestrates the full feature extraction pipeline for one video/clip:
    - Reads first 2 frames
    - Calculates color, bubble size, and texture from the latest frame
    - Calculates speed by comparing the two frames

    Returns a dictionary matching our database's froth_features table.
    """
    frames = get_frames_from_video(video_path, num_frames=2)

    if len(frames) < 2:
        raise ValueError(
            f"Need at least 2 frames to calculate froth speed, "
            f"but only got {len(frames)} from {video_path}"
        )

    latest_frame = frames[-1]

    features = {
        "color_hue_avg": extract_color_feature(latest_frame),
        "avg_bubble_size": extract_bubble_size(latest_frame),
        "texture_score": extract_texture_score(latest_frame),
        "froth_speed": extract_froth_speed(frames[0], frames[1]),
    }

    return features


if __name__ == "__main__":
    from pathlib import Path

    # Scan the videos folder for all .mp4 files, instead of needing one exact filename
    videos_folder = Path("videos")
    video_files = list(videos_folder.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {videos_folder.resolve()}")
    else:
        print(f"Found {len(video_files)} video(s). Testing feature extraction on each:\n")

        for video_path in video_files:
            print(f"--- {video_path.name} ---")
            try:
                result = extract_all_features(video_path)
                for key, value in result.items():
                    print(f"  {key}: {value:.4f}")
            except Exception as e:
                print(f"  Error during feature extraction: {e}")
            print()  # blank line between videos
"""
Image Dataset Processor for Hand Gesture Recognition.
Processes a directory of hand gesture images (e.g. 100 images across 5 classes),
extracts 21 3D landmarks per image using MediaPipe, applies invariant normalization,
and saves the structured dataset to data/gestures_dataset.csv.
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hand_detector import HandDetector
from src.preprocessor import (
    GESTURE_CLASSES,
    FEATURE_DIM,
    NUM_LANDMARKS,
    normalize_landmarks,
    extract_landmarks_from_mediapipe
)

IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "gestures_dataset.csv")


def ensure_image_directories(base_dir: str = IMAGES_DIR) -> Dict[str, str]:
    """Creates subdirectories for each of the 5 gesture classes if not existing."""
    paths = {}
    for gesture in GESTURE_CLASSES:
        p = os.path.join(base_dir, gesture)
        os.makedirs(p, exist_ok=True)
        paths[gesture] = p
    return paths


def process_image_directory(
    images_dir: str = IMAGES_DIR,
    output_csv: str = DEFAULT_CSV_PATH
) -> Dict[str, Any]:
    """
    Scans data/images/<Gesture_Name>/*.(jpg|png), detects hand landmarks with MediaPipe,
    normalizes features, and exports to CSV.

    Returns:
        Summary dictionary with counts of detected images and rejected images.
    """
    ensure_image_directories(images_dir)
    detector = HandDetector()

    feature_cols = []
    for i in range(NUM_LANDMARKS):
        feature_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    columns = feature_cols + ["gesture", "label"]

    rows = []
    stats = {
        "total_images": 0,
        "successful_detections": 0,
        "failed_detections": 0,
        "by_class": {g: 0 for g in GESTURE_CLASSES}
    }

    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    print(f"[*] Scanning image dataset in: {images_dir}")

    for class_idx, gesture_name in enumerate(GESTURE_CLASSES):
        class_folder = os.path.join(images_dir, gesture_name)
        if not os.path.exists(class_folder):
            continue

        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(class_folder, ext)))
            image_files.extend(glob.glob(os.path.join(class_folder, ext.upper())))

        # Deduplicate
        image_files = sorted(list(set(image_files)))

        print(f"  -> Found {len(image_files)} images for class '{gesture_name}'")

        for img_path in image_files:
            stats["total_images"] += 1
            frame = cv2.imread(img_path)

            if frame is None:
                stats["failed_detections"] += 1
                continue

            landmarks, _ = detector.detect(frame)

            if landmarks is None or len(landmarks) == 0:
                stats["failed_detections"] += 1
                continue

            # Extract raw coords & normalize
            raw_coords = extract_landmarks_from_mediapipe(landmarks)
            norm_features = normalize_landmarks(raw_coords)

            record = list(norm_features)
            record.append(gesture_name)
            record.append(class_idx)
            rows.append(record)

            stats["successful_detections"] += 1
            stats["by_class"][gesture_name] += 1

    detector.close()

    if len(rows) == 0:
        print("[!] No hands were detected across the image files.")
        return stats

    # Save to CSV
    df = pd.DataFrame(rows, columns=columns)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[+] Dataset saved to: {output_csv}")
    print(f"    Total processed: {stats['total_images']} images")
    print(f"    Successful samples: {stats['successful_detections']}")
    print(f"    Failed detections: {stats['failed_detections']}")

    return stats


if __name__ == "__main__":
    process_image_directory()

"""
Feature Preprocessor for Hand Gesture Recognition.
Provides strictly identical normalization logic for training data and live webcam inference:
- Translation invariance (relative to wrist landmark 0)
- Scale invariance (normalized by maximum hand span)
- Dimensionality: 21 landmarks * 3 coordinates (x, y, z) = 63 features
"""

import numpy as np
from typing import List, Union, Any

# Standard 32 Gesture Classes
GESTURE_CLASSES = [
    "Open Palm",
    "Fist",
    "Thumbs Up",
    "Thumbs Down",
    "Peace Sign",
    "Pointing Up",
    "OK Sign",
    "Rock Sign",
    "Call Me",
    "Three Fingers",
    "Four Fingers",
    "Number 0",
    "Number 1",
    "Number 2",
    "Number 3",
    "Number 4",
    "Number 5",
    "Gun Sign",
    "Love You Sign",
    "Spock Salute",
    "Pinky Up",
    "Pinch",
    "Crossed Fingers",
    "Stop Sign",
    "Claw Hand",
    "Pointing Forward",
    "Thumb Left",
    "Thumb Right",
    "Two Fingers Down",
    "Salute",
    "Ring Extension",
    "C-Shape"
]

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63 features


def extract_landmarks_from_mediapipe(mediapipe_landmarks: Any) -> np.ndarray:
    """
    Extracts raw (x, y, z) coordinates from MediaPipe hand landmarks.
    Works with both MediaPipe Tasks (NormalizedLandmark objects) and Legacy Solutions.
    Returns:
        np.ndarray of shape (21, 3)
    """
    coords = []
    for lm in mediapipe_landmarks:
        # Some versions provide lm.x, lm.y, lm.z
        coords.append([float(lm.x), float(lm.y), float(lm.z)])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw_coords: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    """
    Normalizes hand landmarks to achieve translation and scale invariance.

    Steps:
    1. Reshape to (21, 3)
    2. Subtract wrist coordinate (landmark 0) from all landmarks (translation invariance)
    3. Calculate maximum Euclidean distance from wrist to any joint (scale factor)
    4. Divide relative coordinates by scale factor (scale invariance)
    5. Flatten to 1D vector of length 63

    Args:
        raw_coords: Shape (21, 3) or flat array of length 63.

    Returns:
        np.ndarray of shape (63,) with dtype float32.
    """
    arr = np.asarray(raw_coords, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != FEATURE_DIM:
            raise ValueError(f"Expected 1D array of size {FEATURE_DIM}, got {arr.shape[0]}")
        arr = arr.reshape((NUM_LANDMARKS, 3))
    elif arr.ndim == 2:
        if arr.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"Expected 2D array of shape (21, 3), got {arr.shape}")
    else:
        raise ValueError(f"Invalid input dimensions: {arr.ndim}. Expected 1D (63,) or 2D (21, 3).")

    # Step 1: Translation relative to wrist (landmark 0)
    wrist = arr[0]  # [x0, y0, z0]
    relative_coords = arr - wrist

    # Step 2: Scale invariance (maximum Euclidean distance from wrist)
    distances = np.linalg.norm(relative_coords, axis=1)
    max_dist = np.max(distances)

    if max_dist < 1e-6:
        # Fallback to prevent divide-by-zero
        scale = 1.0
    else:
        scale = max_dist

    normalized = relative_coords / scale
    return normalized.flatten().astype(np.float32)


def batch_normalize(raw_coords_batch: np.ndarray) -> np.ndarray:
    """
    Normalizes a batch of samples.
    Args:
        raw_coords_batch: Array of shape (N, 63) or (N, 21, 3)
    Returns:
        np.ndarray of shape (N, 63)
    """
    batch = np.asarray(raw_coords_batch, dtype=np.float32)
    n_samples = batch.shape[0]
    normalized_batch = np.zeros((n_samples, FEATURE_DIM), dtype=np.float32)

    for i in range(n_samples):
        sample = batch[i]
        normalized_batch[i] = normalize_landmarks(sample)

    return normalized_batch

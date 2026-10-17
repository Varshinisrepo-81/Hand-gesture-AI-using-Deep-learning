"""
Gesture Prediction Engine for Hand Gesture Recognition.
Connects landmark preprocessor with the trained TensorFlow/Keras deep learning model.
Handles class mapping, probability distribution, and low-confidence filtering.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import normalize_landmarks, extract_landmarks_from_mediapipe, GESTURE_CLASSES
from src.model import load_gesture_model, DEFAULT_MODEL_PATH

DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.json")


class GesturePredictor:
    """Performs real-time gesture inference given hand landmarks."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, label_path: str = DEFAULT_LABEL_PATH):
        self.model_path = model_path
        self.label_path = label_path
        self.model = None
        self.label_map = {i: name for i, name in enumerate(GESTURE_CLASSES)}
        self.load_model_and_labels()

    def is_model_loaded(self) -> bool:
        """Returns True if the deep learning model is successfully loaded."""
        return self.model is not None

    def load_model_and_labels(self) -> bool:
        """Loads or reloads the trained Keras model and label mapping."""
        if os.path.exists(self.model_path):
            try:
                self.model = load_gesture_model(self.model_path)
            except Exception as e:
                print(f"[!] Error loading model: {e}")
                self.model = None

        if os.path.exists(self.label_path):
            try:
                with open(self.label_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                    self.label_map = {int(k): v for k, v in raw_map.items()}
            except Exception as e:
                print(f"[!] Error loading label mapping: {e}")

        return self.model is not None

    def predict(
        self,
        landmarks: Any,
        confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Runs neural network inference on raw hand landmarks.

        Args:
            landmarks: MediaPipe landmark list or numpy coordinate array (21, 3).
            confidence_threshold: Threshold below which prediction is marked 'Uncertain'.

        Returns:
            Dictionary with prediction results, confidence score, and probabilities.
        """
        if self.model is None:
            return {
                "gesture": "Model Not Loaded",
                "raw_gesture": "None",
                "confidence": 0.0,
                "is_low_confidence": True,
                "probabilities": {},
                "normalized_features": None,
                "error": "No trained model found on disk. Please train the model first."
            }

        # 1. Extract raw coordinates
        if isinstance(landmarks, (list, tuple)) and hasattr(landmarks[0], "x"):
            raw_coords = extract_landmarks_from_mediapipe(landmarks)
        else:
            raw_coords = np.asarray(landmarks, dtype=np.float32)

        # 2. Apply identical preprocessing
        norm_features = normalize_landmarks(raw_coords)

        # 3. Model inference (Batch dimension: 1, 63)
        input_tensor = np.expand_dims(norm_features, axis=0)
        probs = self.model.predict(input_tensor, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_gesture = self.label_map.get(pred_idx, f"Class {pred_idx}")

        # 4. Low-confidence handling
        is_low_conf = confidence < confidence_threshold
        if is_low_conf:
            gesture_display = "Uncertain / Low Confidence"
        else:
            gesture_display = raw_gesture

        # Build full probability dictionary
        prob_dict = {
            self.label_map.get(i, f"Class {i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        return {
            "gesture": gesture_display,
            "raw_gesture": raw_gesture,
            "confidence": round(confidence, 4),
            "is_low_confidence": is_low_conf,
            "probabilities": prob_dict,
            "normalized_features": norm_features,
            "error": None
        }

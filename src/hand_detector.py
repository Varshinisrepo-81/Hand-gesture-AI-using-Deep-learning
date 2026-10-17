"""
MediaPipe Hand Landmark Detector Module.
Uses Google MediaPipe's HandLandmarker Task to extract 21 3D landmarks in real time.
Includes automated model asset retrieval and custom visual styling for joint skeletons.
"""

import os
import urllib.request
import numpy as np
import cv2
from typing import Optional, List, Tuple, Any

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# 21 Hand Skeleton Connection Pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}


def ensure_task_model_exists(model_path: str = TASK_MODEL_PATH) -> str:
    """Verifies that MediaPipe task file exists, downloading it if necessary."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[*] Downloading MediaPipe Hand Landmarker model to {model_path}...")
        try:
            req = urllib.request.Request(TASK_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(model_path, "wb") as f:
                f.write(resp.read())
            print("[+] Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe Hand Landmarker model: {e}")
    return model_path


class HandDetector:
    """Encapsulates MediaPipe HandLandmarker detection and drawing."""

    def __init__(self, num_hands: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        model_path = ensure_task_model_exists()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Any]], Optional[str]]:
        """
        Processes a BGR video frame and returns extracted landmarks and handedness.

        Returns:
            Tuple of (landmarks_list, handedness_str)
            If no hand detected, returns (None, None).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None, None

        landmarks = result.hand_landmarks[0]
        handedness = "Hand"
        if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
            handedness = result.handedness[0][0].display_name or result.handedness[0][0].category_name or "Hand"

        return landmarks, handedness

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: List[Any],
        handedness: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws high-contrast neon skeleton joints and connections on the frame.
        """
        if landmarks is None or len(landmarks) == 0:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        # Calculate pixel coordinates
        pts_2d = []
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            pts_2d.append((cx, cy))

        # 1. Draw connection bones (Cyan neon)
        bone_color = (255, 200, 0)  # BGR Cyan/Blue-Green
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pts_2d[start_idx]
            pt2 = pts_2d[end_idx]
            cv2.line(annotated, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # 2. Draw joints (Green/lime with white centers)
        for idx, (cx, cy) in enumerate(pts_2d):
            if idx in FINGERTIP_INDICES:
                # Fingertips: vibrant magenta/orange glow
                cv2.circle(annotated, (cx, cy), 7, (0, 140, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            elif idx == 0:
                # Wrist: Cyan anchor
                cv2.circle(annotated, (cx, cy), 8, (255, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Intermediate joints: Emerald green
                cv2.circle(annotated, (cx, cy), 5, (0, 230, 115), -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Draw bounding box with padding
        x_coords = [p[0] for p in pts_2d]
        y_coords = [p[1] for p in pts_2d]
        min_x = max(0, min(x_coords) - 20)
        max_x = min(w, max(x_coords) + 20)
        min_y = max(0, min(y_coords) - 20)
        max_y = min(h, max(y_coords) + 20)

        # Sleek bounding box outline
        cv2.rectangle(annotated, (min_x, min_y), (max_x, max_y), (100, 100, 255), 1, cv2.LINE_AA)

        # Handedness badge
        if handedness:
            badge_text = f"{handedness} Hand"
            cv2.putText(
                annotated,
                badge_text,
                (min_x, max(15, min_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 220, 255),
                1,
                cv2.LINE_AA
            )

        return annotated

    def draw_hud_overlay(
        self,
        frame_bgr: np.ndarray,
        gesture: str,
        confidence: float,
        is_low_conf: bool = False,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Draws a semi-transparent HUD banner with prediction and confidence score."""
        if frame_bgr is None:
            return None
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        # Semi-transparent top card overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (15, 15), (min(w - 15, 340), 95), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (15, 15), (min(w - 15, 340), 95), (70, 80, 100), 1, cv2.LINE_AA)

        # Status text & color
        if is_low_conf:
            color = (0, 165, 255)  # Amber / Orange
            status_text = f"{gesture}"
        elif gesture == "No Hand Detected":
            color = (130, 140, 150)  # Gray
            status_text = gesture
        else:
            color = (0, 230, 115)  # Emerald Green
            status_text = gesture

        # Gesture name
        cv2.putText(annotated, status_text, (28, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)

        # Confidence percentage
        conf_pct = int(confidence * 100)
        cv2.putText(annotated, f"Conf: {conf_pct}%", (28, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 225, 230), 1, cv2.LINE_AA)

        # Mini confidence progress bar
        bar_x = 135
        bar_y = 67
        bar_w = 175
        bar_h = 10
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 55, 65), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        if fill_w > 0:
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # FPS indicator in top-right
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (w - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 180), 1, cv2.LINE_AA)

        return annotated

    def close(self):
        """Releases detector resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()

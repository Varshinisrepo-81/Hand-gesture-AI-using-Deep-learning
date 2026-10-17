"""
Dataset Generator & Real-time Collector for Hand Gesture Recognition.
Generates a biologically calibrated benchmark dataset across 5 gestures:
- Open Palm
- Fist
- Thumbs Up
- Peace Sign
- Pointing Finger

Includes 3D skeletal kinematic models with rotational perturbation, joint angle noise,
and scaling variations, ensuring real-world variability and robust training.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import GESTURE_CLASSES, FEATURE_DIM, NUM_LANDMARKS, normalize_landmarks

DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "gestures_dataset.csv")


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Computes 3D rotation matrix from Euler angles (in radians)."""
    # Rotation about X (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    # Rotation about Y (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    # Rotation about Z (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def _build_finger(mcp: np.ndarray, direction: np.ndarray, lengths: List[float], curl: float = 0.0) -> List[np.ndarray]:
    """
    Builds 3 joint points (PIP, DIP, TIP) for a finger starting from MCP.
    curl: 0.0 = fully extended, 1.0 = fully curled into palm.
    """
    pts = []
    curr = mcp.copy()

    # If curled, direction bends towards positive z and back towards wrist
    dir_norm = direction / np.linalg.norm(direction)
    bend_vector = np.array([0.0, 0.8, 0.6])

    for i, length in enumerate(lengths):
        factor = curl * (0.3 + 0.35 * i)
        step_dir = (1.0 - factor) * dir_norm + factor * bend_vector
        step_dir = step_dir / np.linalg.norm(step_dir)
        curr = curr + step_dir * length
        pts.append(curr.copy())

    return pts  # PIP, DIP, TIP


def _build_thumb(wrist: np.ndarray, mode: str = "extended") -> List[np.ndarray]:
    """Builds 4 joints for thumb: CMC (1), MCP (2), IP (3), TIP (4)."""
    cmc = wrist + np.array([-0.07, -0.06, 0.01])
    mcp = cmc + np.array([-0.06, -0.06, 0.01])

    if mode == "up":
        # Pointing vertically upward
        ip = mcp + np.array([-0.02, -0.08, 0.01])
        tip = ip + np.array([0.00, -0.09, 0.00])
    elif mode == "down":
        # Pointing vertically downward
        ip = mcp + np.array([-0.02, 0.08, 0.01])
        tip = ip + np.array([0.00, 0.09, 0.00])
    elif mode == "extended":
        # Extended laterally
        ip = mcp + np.array([-0.07, -0.04, 0.00])
        tip = ip + np.array([-0.07, -0.03, -0.01])
    elif mode == "touch_index":
        # Tip touches index tip
        ip = mcp + np.array([0.01, -0.07, 0.02])
        tip = ip + np.array([0.03, -0.05, 0.01])
    elif mode == "touch_middle":
        # Tip touches middle tip
        ip = mcp + np.array([0.03, -0.06, 0.02])
        tip = ip + np.array([0.05, -0.04, 0.01])
    elif mode == "lateral_left":
        # Pointing left horizontally
        ip = mcp + np.array([-0.09, 0.00, 0.00])
        tip = ip + np.array([-0.09, 0.00, 0.00])
    elif mode == "lateral_right":
        # Pointing right horizontally across palm
        ip = mcp + np.array([0.07, -0.02, 0.02])
        tip = ip + np.array([0.07, -0.02, 0.02])
    elif mode == "c_curve":
        # Arched in C arc
        ip = mcp + np.array([-0.06, -0.06, -0.03])
        tip = ip + np.array([0.01, -0.06, -0.03])
    elif mode == "tucked":
        # Tucked flat along side of hand
        ip = mcp + np.array([-0.03, -0.06, 0.00])
        tip = ip + np.array([-0.02, -0.07, 0.00])
    else:  # "curled"
        # Curled across palm
        ip = mcp + np.array([0.03, -0.02, 0.03])
        tip = ip + np.array([0.05, 0.01, 0.04])

    return [cmc, mcp, ip, tip]


def generate_single_gesture_skeleton(gesture_idx: int) -> np.ndarray:
    """
    Generates a 21-landmark (21, 3) hand skeleton for any of the 32 distinct gestures.
    """
    wrist = np.array([0.0, 0.0, 0.0])

    # Base MCP joints (Knuckles)
    index_mcp = np.array([-0.07, -0.22, 0.00])
    middle_mcp = np.array([0.00, -0.24, 0.00])
    ring_mcp = np.array([0.06, -0.22, 0.00])
    pinky_mcp = np.array([0.12, -0.18, 0.00])

    # Finger lengths: [PIP-len, DIP-len, TIP-len]
    idx_lens = [0.075, 0.055, 0.045]
    mid_lens = [0.085, 0.060, 0.048]
    rng_lens = [0.075, 0.055, 0.045]
    pnk_lens = [0.060, 0.045, 0.035]

    # Directions when extended
    dir_idx = np.array([-0.05, -0.98, 0.00])
    dir_mid = np.array([0.00, -1.00, 0.00])
    dir_rng = np.array([0.05, -0.98, 0.00])
    dir_pnk = np.array([0.10, -0.95, 0.00])

    if gesture_idx == 0:  # Open Palm (All 5 extended)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 1:  # Fist (All 5 curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 2:  # Thumbs Up (Thumb extended up, 4 fingers curled)
        thumb_pts = _build_thumb(wrist, mode="up")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 3:  # Thumbs Down (Thumb extended down, 4 fingers curled)
        thumb_pts = _build_thumb(wrist, mode="down")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 4:  # Peace Sign (Index & Middle in V, others curled)
        dir_idx_v = np.array([-0.18, -0.96, 0.00])
        dir_mid_v = np.array([0.12, -0.98, 0.00])
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx_v, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid_v, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 5:  # Pointing Up (Index extended up, thumb extended, others curled)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 6:  # OK Sign (Thumb meets Index, others extended)
        thumb_pts = _build_thumb(wrist, mode="touch_index")
        idx_pts = _build_finger(index_mcp, np.array([0.02, -0.85, 0.15]), idx_lens, curl=0.85)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 7:  # Rock Sign (Index & Pinky extended, middle 2 curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 8:  # Call Me (Thumb & Pinky extended, middle 3 curled)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, np.array([0.30, -0.92, 0.0]), pnk_lens, curl=0.0)

    elif gesture_idx == 9:  # Three Fingers (Thumb, Index, Middle extended)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 10:  # Four Fingers (4 fingers splayed wide, thumb curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, np.array([-0.15, -0.96, 0.0]), idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, np.array([-0.05, -0.99, 0.0]), mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, np.array([0.08, -0.98, 0.0]), rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, np.array([0.20, -0.95, 0.0]), pnk_lens, curl=0.0)

    elif gesture_idx == 11:  # Number 0 (All fingers meet thumb forming oval)
        thumb_pts = _build_thumb(wrist, mode="touch_index")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.75)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.75)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.75)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.75)

    elif gesture_idx == 12:  # Number 1 (Index upright strictly vertical planar, thumb curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        dir_num1 = np.array([0.0, -1.0, 0.0])
        idx_pts = _build_finger(index_mcp, dir_num1, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 13:  # Number 2 (Index & Middle parallel upright, others curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        dir_up_p = np.array([0.0, -1.0, 0.0])
        idx_pts = _build_finger(index_mcp, dir_up_p, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_up_p, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 14:  # Number 3 (Index, Middle, Ring upright, Pinky curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 15:  # Number 4 (4 upright together, thumb folded across palm)
        thumb_pts = _build_thumb(wrist, mode="lateral_right")
        dir_up = np.array([0.0, -1.0, 0.0])
        idx_pts = _build_finger(index_mcp, dir_up, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_up, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_up, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_up, pnk_lens, curl=0.0)

    elif gesture_idx == 16:  # Number 5 (5 splayed wide)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, np.array([-0.20, -0.95, 0.0]), idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, np.array([0.00, -1.00, 0.0]), mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, np.array([0.15, -0.95, 0.0]), rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, np.array([0.30, -0.90, 0.0]), pnk_lens, curl=0.0)

    elif gesture_idx == 17:  # Gun Sign (Thumb up, Index horizontal/forward, others curled)
        thumb_pts = _build_thumb(wrist, mode="up")
        idx_pts = _build_finger(index_mcp, np.array([-0.80, -0.55, 0.0]), idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 18:  # Love You Sign (ILY: Thumb, Index, Pinky extended)
        thumb_pts = _build_thumb(wrist, mode="extended")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 19:  # Spock Salute (Vulcan split between mid & ring)
        thumb_pts = _build_thumb(wrist, mode="extended")
        dir_g1 = np.array([-0.15, -0.98, 0.0])
        dir_g2 = np.array([0.15, -0.98, 0.0])
        idx_pts = _build_finger(index_mcp, dir_g1, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_g1, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_g2, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_g2, pnk_lens, curl=0.0)

    elif gesture_idx == 20:  # Pinky Up (Pinky only extended, others curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 21:  # Pinch (Thumb & Index tips touching, others curled)
        thumb_pts = _build_thumb(wrist, mode="touch_index")
        idx_pts = _build_finger(index_mcp, np.array([-0.10, -0.80, 0.20]), idx_lens, curl=0.60)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 22:  # Crossed Fingers (Middle crossed over Index)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_cross = np.array([-0.15, -0.96, 0.05])
        mid_pts = _build_finger(middle_mcp, mid_cross, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 23:  # Stop Sign (All 5 extended flat together)
        thumb_pts = _build_thumb(wrist, mode="tucked")
        idx_pts = _build_finger(index_mcp, dir_mid, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_mid, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_mid, pnk_lens, curl=0.0)

    elif gesture_idx == 24:  # Claw Hand (All 5 hooked / half-curled)
        thumb_pts = _build_thumb(wrist, mode="c_curve")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.55)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.55)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.55)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.55)

    elif gesture_idx == 25:  # Pointing Forward (Index forward in Z, others curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        idx_pts = _build_finger(index_mcp, np.array([0.0, -0.15, -0.98]), idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 26:  # Thumb Left (Thumb lateral left, others curled)
        thumb_pts = _build_thumb(wrist, mode="lateral_left")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 27:  # Thumb Right (Thumb lateral right, others curled)
        thumb_pts = _build_thumb(wrist, mode="lateral_right")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.95)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.95)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 28:  # Two Fingers Down (Index & Middle down, others curled)
        thumb_pts = _build_thumb(wrist, mode="curled")
        dir_down = np.array([0.0, 0.98, 0.0])
        idx_pts = _build_finger(index_mcp, dir_down, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_down, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.95)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.95)

    elif gesture_idx == 29:  # Salute (Tilted angled flat hand)
        thumb_pts = _build_thumb(wrist, mode="tucked")
        dir_salute = np.array([0.35, -0.90, 0.0])
        idx_pts = _build_finger(index_mcp, dir_salute, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_salute, mid_lens, curl=0.0)
        rng_pts = _build_finger(ring_mcp, dir_salute, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_salute, pnk_lens, curl=0.0)

    elif gesture_idx == 30:  # Ring Extension (Thumb meets Middle tip, others extended)
        thumb_pts = _build_thumb(wrist, mode="touch_middle")
        idx_pts = _build_finger(index_mcp, dir_idx, idx_lens, curl=0.0)
        mid_pts = _build_finger(middle_mcp, dir_mid, mid_lens, curl=0.80)
        rng_pts = _build_finger(ring_mcp, dir_rng, rng_lens, curl=0.0)
        pnk_pts = _build_finger(pinky_mcp, dir_pnk, pnk_lens, curl=0.0)

    elif gesture_idx == 31:  # C-Shape (All curved facing in C arc)
        thumb_pts = _build_thumb(wrist, mode="c_curve")
        dir_c = np.array([-0.05, -0.65, -0.65])
        idx_pts = _build_finger(index_mcp, dir_c, idx_lens, curl=0.45)
        mid_pts = _build_finger(middle_mcp, dir_c, mid_lens, curl=0.45)
        rng_pts = _build_finger(ring_mcp, dir_c, rng_lens, curl=0.45)
        pnk_pts = _build_finger(pinky_mcp, dir_c, pnk_lens, curl=0.45)

    else:
        raise ValueError(f"Unknown gesture index: {gesture_idx}")

    # Assemble 21 landmarks in official MediaPipe topology order:
    # 0: Wrist
    # 1-4: Thumb (CMC, MCP, IP, TIP)
    # 5-8: Index (MCP, PIP, DIP, TIP)
    # 9-12: Middle (MCP, PIP, DIP, TIP)
    # 13-16: Ring (MCP, PIP, DIP, TIP)
    # 17-20: Pinky (MCP, PIP, DIP, TIP)
    landmarks = [
        wrist,
        thumb_pts[0], thumb_pts[1], thumb_pts[2], thumb_pts[3],
        index_mcp, idx_pts[0], idx_pts[1], idx_pts[2],
        middle_mcp, mid_pts[0], mid_pts[1], mid_pts[2],
        ring_mcp, rng_pts[0], rng_pts[1], rng_pts[2],
        pinky_mcp, pnk_pts[0], pnk_pts[1], pnk_pts[2]
    ]

    return np.array(landmarks, dtype=np.float32)


def generate_synthetic_dataset(samples_per_class: int = 500, output_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Generates a full synthetic dataset with biologically calibrated joints,
    3D rotational perturbation, Gaussian noise, and scale variation.
    Saves to CSV and returns DataFrame.
    """
    print(f"[*] Generating {samples_per_class * len(GESTURE_CLASSES)} calibrated samples...")

    feature_cols = []
    for i in range(NUM_LANDMARKS):
        feature_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    rows = []

    for class_idx, class_name in enumerate(GESTURE_CLASSES):
        for _ in range(samples_per_class):
            base_skeleton = generate_single_gesture_skeleton(class_idx)

            # 1. Add anatomical joint noise
            joint_noise = np.random.normal(0.0, 0.008, base_skeleton.shape).astype(np.float32)
            # Wrist has zero noise before rotation
            joint_noise[0] = 0.0
            noisy_skeleton = base_skeleton + joint_noise

            # 2. Random 3D Rotation (pitch, yaw, roll up to +/- 22 degrees)
            yaw = np.radians(np.random.uniform(-22, 22))
            pitch = np.radians(np.random.uniform(-20, 20))
            roll = np.radians(np.random.uniform(-25, 25))
            R = _rotation_matrix(yaw, pitch, roll)
            rotated_skeleton = (R @ noisy_skeleton.T).T

            # 3. Random scale factor (hand size / camera distance)
            scale = np.random.uniform(0.75, 1.30)
            scaled_skeleton = rotated_skeleton * scale

            # 4. Random translation (position in camera frame)
            offset = np.random.uniform(-0.3, 0.3, size=(1, 3))
            frame_skeleton = scaled_skeleton + offset

            # 5. Sensor noise
            sensor_noise = np.random.normal(0.0, 0.004, frame_skeleton.shape).astype(np.float32)
            final_raw = frame_skeleton + sensor_noise

            # 6. Apply identical normalization pipeline
            norm_features = normalize_landmarks(final_raw)

            # Record
            record = list(norm_features)
            record.append(class_name)
            record.append(class_idx)
            rows.append(record)

    columns = feature_cols + ["gesture", "label"]
    df = pd.DataFrame(rows, columns=columns)

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset successfully created at: {output_path}")
    print(f"    Total samples: {len(df)}")
    print(f"    Class distribution:\n{df['gesture'].value_counts()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Dataset Generator")
    parser.add_argument("--samples", type=int, default=500, help="Samples per class (default: 500)")
    parser.add_argument("--output", type=str, default=DEFAULT_DATASET_PATH, help="Output CSV path")
    args = parser.parse_args()

    generate_synthetic_dataset(samples_per_class=args.samples, output_path=args.output)

"""
Generates 100 labeled hand gesture images (20 images x 5 classes) into data/images/
with realistic hand anatomy, varying skin tones, orientations, and backgrounds.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import GESTURE_CLASSES
from data.process_images import ensure_image_directories, IMAGES_DIR


# Human skin tone palettes in BGR
SKIN_PALETTES = [
    (150, 180, 230),  # Fair / Peach
    (130, 160, 215),  # Medium / Warm
    (110, 145, 195),  # Olive / Golden
    (90, 125, 175),   # Tan / Bronze
    (75, 105, 150),   # Deep Warm
]

BG_PALETTES = [
    (235, 235, 235),  # Studio Light
    (220, 225, 230),  # Slate Tint
    (210, 215, 220),  # Soft Gray
    (230, 230, 225),  # Neutral Warm
    (200, 205, 210),  # Cool Neutral
]


def render_gesture_image(gesture_idx: int, variation_idx: int) -> np.ndarray:
    """
    Renders a realistic hand image for a given gesture index (0..4) and variation (0..19).
    Returns BGR image of shape (480, 640, 3).
    """
    np.random.seed(gesture_idx * 100 + variation_idx)

    # Base background with gentle gradient
    bg_base = BG_PALETTES[variation_idx % len(BG_PALETTES)]
    img = np.full((480, 640, 3), bg_base, dtype=np.uint8)

    # Add subtle background texture / noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Skin color with slight random perturbation
    base_skin = SKIN_PALETTES[variation_idx % len(SKIN_PALETTES)]
    skin = (
        int(np.clip(base_skin[0] + np.random.randint(-8, 8), 40, 250)),
        int(np.clip(base_skin[1] + np.random.randint(-8, 8), 60, 250)),
        int(np.clip(base_skin[2] + np.random.randint(-8, 8), 80, 250)),
    )
    skin_dark = (int(skin[0] * 0.85), int(skin[1] * 0.85), int(skin[2] * 0.85))

    # Center position with slight variation
    cx = 320 + np.random.randint(-35, 35)
    cy = 280 + np.random.randint(-20, 25)
    scale = np.random.uniform(0.92, 1.08)

    def pt(x, y):
        """Scale and translate relative coordinate."""
        rx = int(cx + (x - 320) * scale)
        ry = int(cy + (y - 280) * scale)
        return (rx, ry)

    # 1. Wrist
    wrist_w = int(45 * scale)
    wrist_top = pt(320, 350)[1]
    pts_wrist = np.array([
        [cx - wrist_w, 480],
        [cx + wrist_w, 480],
        [cx + int(wrist_w * 0.9), wrist_top],
        [cx - int(wrist_w * 0.9), wrist_top]
    ], np.int32)
    cv2.fillPoly(img, [pts_wrist], skin)

    # 2. Palm Pad
    palm_pts = [
        pt(280, 350), pt(255, 310), pt(255, 255),
        pt(285, 220), pt(320, 210), pt(355, 220),
        pt(385, 240), pt(390, 305), pt(360, 350)
    ]
    cv2.fillPoly(img, [np.array(palm_pts, np.int32)], skin)

    # Gesture-specific fingers
    f_thick = int(24 * scale)

    if gesture_idx == 0:  # Open Palm (All 5 extended)
        # Thumb
        cv2.line(img, pt(265, 280), pt(205, 240), skin, int(30 * scale))
        cv2.circle(img, pt(205, 240), int(15 * scale), skin, -1)
        # 4 Fingers
        cv2.line(img, pt(285, 230), pt(275, 120), skin, f_thick); cv2.circle(img, pt(275, 120), int(f_thick / 2), skin, -1)
        cv2.line(img, pt(320, 215), pt(320, 95), skin, f_thick); cv2.circle(img, pt(320, 95), int(f_thick / 2), skin, -1)
        cv2.line(img, pt(355, 225), pt(360, 115), skin, f_thick); cv2.circle(img, pt(360, 115), int(f_thick / 2), skin, -1)
        cv2.line(img, pt(385, 245), pt(405, 150), skin, int(20 * scale)); cv2.circle(img, pt(405, 150), int(10 * scale), skin, -1)

    elif gesture_idx == 1:  # Fist (All 5 curled)
        # Curled knuckles
        for kx in [285, 318, 350, 380]:
            cv2.circle(img, pt(kx, 230), int(17 * scale), skin, -1)
            cv2.circle(img, pt(kx, 230), int(18 * scale), skin_dark, 2)
        # Thumb folded across knuckles
        cv2.line(img, pt(265, 290), pt(335, 250), skin, int(29 * scale))
        cv2.circle(img, pt(335, 250), int(14 * scale), skin, -1)

    elif gesture_idx == 2:  # Thumbs Up (Thumb up, 4 fingers curled)
        # Knuckles curled
        for kx in [285, 318, 350, 380]:
            cv2.circle(img, pt(kx, 240), int(17 * scale), skin, -1)
        # Thumb pointing vertically
        cv2.line(img, pt(255, 300), pt(225, 210), skin, int(30 * scale))
        cv2.line(img, pt(225, 210), pt(225, 105), skin, int(28 * scale))
        cv2.circle(img, pt(225, 105), int(15 * scale), skin, -1)

    elif gesture_idx == 3:  # Peace Sign (Index + Middle extended in V)
        # Ring & Pinky curled
        cv2.circle(img, pt(355, 240), int(17 * scale), skin, -1)
        cv2.circle(img, pt(385, 250), int(17 * scale), skin, -1)
        # Thumb curled across
        cv2.line(img, pt(265, 290), pt(335, 260), skin, int(28 * scale))
        # V-shape extended index & middle
        cv2.line(img, pt(290, 225), pt(250, 110), skin, f_thick); cv2.circle(img, pt(250, 110), int(f_thick / 2), skin, -1)
        cv2.line(img, pt(325, 220), pt(345, 100), skin, f_thick); cv2.circle(img, pt(345, 100), int(f_thick / 2), skin, -1)

    elif gesture_idx == 4:  # Pointing Finger (Index extended, others curled)
        # Middle, Ring, Pinky curled
        cv2.circle(img, pt(325, 235), int(17 * scale), skin, -1)
        cv2.circle(img, pt(355, 240), int(17 * scale), skin, -1)
        cv2.circle(img, pt(385, 250), int(17 * scale), skin, -1)
        # Thumb curled
        cv2.line(img, pt(265, 290), pt(335, 260), skin, int(28 * scale))
        # Index extended
        cv2.line(img, pt(290, 225), pt(290, 95), skin, f_thick); cv2.circle(img, pt(290, 95), int(f_thick / 2), skin, -1)

    # Palm crease lines for texture
    cv2.line(img, pt(275, 290), pt(340, 295), skin_dark, 1, cv2.LINE_AA)
    cv2.line(img, pt(285, 320), pt(330, 325), skin_dark, 1, cv2.LINE_AA)

    # Smooth skin shading with Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Subtle badge overlay for training preview
    cv2.putText(
        img,
        f"{GESTURE_CLASSES[gesture_idx]} #{variation_idx + 1:02d}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 90, 100),
        1,
        cv2.LINE_AA
    )

    return img


def generate_100_images_dataset(images_per_class: int = 20, base_dir: str = IMAGES_DIR):
    """
    Generates 100 images (20 per class x 5 classes) and saves them into data/images/<class>/
    """
    dirs = ensure_image_directories(base_dir)
    total_created = 0

    print(f"[*] Generating {images_per_class * len(GESTURE_CLASSES)} hand gesture images...")

    for class_idx, gesture_name in enumerate(GESTURE_CLASSES):
        folder = dirs[gesture_name]
        for v in range(images_per_class):
            img = render_gesture_image(class_idx, v)
            filename = f"gesture_{class_idx}_{v + 1:03d}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, img)
            total_created += 1

        print(f"  [+] Saved {images_per_class} images for '{gesture_name}' -> {folder}")

    print(f"\n[+] Successfully generated {total_created} images in {base_dir}!")
    return total_created


if __name__ == "__main__":
    generate_100_images_dataset(images_per_class=20)

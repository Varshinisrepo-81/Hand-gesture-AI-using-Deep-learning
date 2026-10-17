"""
Webcam Image Dataset Capture Utility for Hand Gesture Recognition.
Captures real photos from your webcam for each gesture class (e.g. 20 photos x 5 classes = 100 photos).
Saves images directly to data/images/<Gesture_Name>/image_XXX.jpg and updates the training dataset.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.camera import CameraManager
from src.hand_detector import HandDetector
from src.preprocessor import GESTURE_CLASSES
from data.process_images import ensure_image_directories, process_image_directory, IMAGES_DIR


def capture_gesture_images(
    gesture_name: str,
    target_count: int = 20,
    device_index: int = 0,
    auto_burst: bool = False
) -> int:
    """
    Opens webcam and captures photos for a specific gesture class.

    Controls in OpenCV window:
        's' or SPACE: Snap single photo
        'a': Toggle auto-burst mode (captures photo every 0.6 seconds when hand is visible)
        'q' or ESC: Finish and return
    """
    class_dir = os.path.join(IMAGES_DIR, gesture_name)
    os.makedirs(class_dir, exist_ok=True)

    # Count existing images
    existing = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    captured_count = 0

    cam = CameraManager(device_index=device_index, width=1280, height=720)
    cam.set_enhancements(enable_clahe=True, enable_sharpening=True, enable_denoising=False)
    success, msg = cam.start()
    if not success:
        print(f"[!] Camera Error: {msg}")
        return 0

    detector = HandDetector()

    print("\n" + "="*60)
    print(f" CAPTURING IMAGES FOR GESTURE: {gesture_name.upper()}")
    print("="*60)
    print(f" Target count: {target_count} photos (Currently existing: {existing})")
    print(f" Stream Quality: {cam.get_resolution_string()}")
    print(" Controls:")
    print("   [SPACE] or [S] : Snap photo")
    print("   [A]            : Toggle automatic burst capture (0.6s interval)")
    print("   [Q] or [ESC]   : Finish and save")
    print("="*60 + "\n")

    last_snap_time = 0.0
    is_burst_active = auto_burst

    window_name = f"Capture Dataset: {gesture_name} - Hand Gesture AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)

    try:
        while captured_count < target_count:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                break

            h, w, _ = frame.shape
            display_frame = frame.copy()

            # Detect hand
            landmarks, handedness = detector.detect(frame)
            hand_in_view = landmarks is not None

            # Draw skeleton for visual feedback
            if hand_in_view:
                display_frame = detector.draw_landmarks(display_frame, landmarks, handedness)

            # Top UI banner
            banner_bg = (30, 30, 40)
            cv2.rectangle(display_frame, (10, 10), (w - 10, 80), banner_bg, -1)
            cv2.rectangle(display_frame, (10, 10), (w - 10, 80), (80, 80, 100), 1)

            # Status text
            status_color = (0, 230, 115) if hand_in_view else (0, 165, 255)
            status_msg = "Hand Detected - Ready to Snap!" if hand_in_view else "Position your hand in frame..."

            cv2.putText(
                display_frame,
                f"Gesture: {gesture_name} | Saved: {captured_count}/{target_count}",
                (25, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                display_frame,
                f"Status: {status_msg} {'[BURST ON]' if is_burst_active else ''}",
                (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                status_color,
                1,
                cv2.LINE_AA
            )

            # Auto-burst logic
            curr_time = time.time()
            trigger_snap = False

            if is_burst_active and hand_in_view and (curr_time - last_snap_time > 0.6):
                trigger_snap = True

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                print("[*] Capture stopped by user.")
                break
            elif key in (ord('s'), 32):  # 's' or SPACE
                if hand_in_view:
                    trigger_snap = True
                else:
                    print("[!] No hand detected in frame. Move hand into view to snap.")
            elif key == ord('a'):
                is_burst_active = not is_burst_active
                print(f"[*] Burst mode: {'ENABLED' if is_burst_active else 'DISABLED'}")

            # Save image
            if trigger_snap:
                img_num = existing + captured_count + 1
                img_filename = f"image_{img_num:04d}.jpg"
                img_path = os.path.join(class_dir, img_filename)

                # Save raw BGR frame
                cv2.imwrite(img_path, frame)
                captured_count += 1
                last_snap_time = curr_time

                # Flash green border on frame
                cv2.rectangle(display_frame, (0, 0), (w, h), (0, 255, 0), 10)
                print(f"[+] Saved ({captured_count}/{target_count}): {img_path}")

            cv2.imshow(window_name, display_frame)

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()

    print(f"\n[+] Completed capture for '{gesture_name}': {captured_count} new images saved to {class_dir}")
    return captured_count


def get_missing_classes(min_images: int = 10, base_dir: str = IMAGES_DIR) -> List[Tuple[str, int]]:
    """
    Scans data/images/ and returns a list of (gesture_name, current_count)
    for classes with fewer than min_images.
    """
    ensure_image_directories(base_dir)
    missing = []
    for g in GESTURE_CLASSES:
        class_folder = os.path.join(base_dir, g)
        count = len([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if count < min_images:
            missing.append((g, count))
    return missing


def print_dataset_status(min_images: int = 10, base_dir: str = IMAGES_DIR):
    """Prints a clear terminal status breakdown of all 32 gesture classes."""
    ensure_image_directories(base_dir)
    print("\n" + "="*65)
    print("      HAND GESTURE RECOGNITION - DATASET STATUS AUDIT")
    print("="*65)
    print(f"{'#':<4} {'Gesture Name':<28} {'Images':<10} {'Status':<15}")
    print("-" * 65)

    ready_count = 0
    missing_count = 0

    for idx, g in enumerate(GESTURE_CLASSES, 1):
        class_folder = os.path.join(base_dir, g)
        count = len([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        status = "READY" if count >= min_images else ("PARTIAL" if count > 0 else "MISSING")
        if count >= min_images:
            ready_count += 1
        else:
            missing_count += 1
        print(f"{idx:<4} {g:<28} {count:<10} {status:<15}")

    print("=" * 65)
    print(f"Total Gestures: {len(GESTURE_CLASSES)} | Ready: {ready_count} | Missing/Under-sampled: {missing_count}")
    print("=" * 65 + "\n")


def capture_dataset_workflow(target_classes: List[str], images_per_class: int = 20, device_index: int = 0):
    """
    Guides the user through capturing images for a specific list of gesture classes.
    """
    print("\n" + "#"*65)
    print(f"  HAND GESTURE DATASET COLLECTOR ({len(target_classes)} TARGET CLASSES)")
    print("#"*65)
    print(f"Target count: {images_per_class} images per class for {len(target_classes)} classes.\n")

    total_captured = 0
    for idx, gesture in enumerate(target_classes, 1):
        print(f"\n--- Progress: [{idx}/{len(target_classes)}] Gesture: {gesture} ---")
        input(f"Press ENTER when ready to capture photos for '{gesture}' (or Ctrl+C to cancel)...")
        count = capture_gesture_images(
            gesture_name=gesture,
            target_count=images_per_class,
            device_index=device_index,
            auto_burst=False
        )
        total_captured += count

    print("\n" + "="*65)
    print(f" CAPTURE WORKFLOW FINISHED! Total photos captured: {total_captured}")
    print("="*65)

    if total_captured > 0:
        print("\n[*] Processing newly captured images into CSV landmark dataset...")
        process_image_directory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Hand Gesture Images via Webcam (30+ Classes)")
    parser.add_argument("--count", type=int, default=20, help="Images to capture per class (default: 20)")
    parser.add_argument("--gesture", type=str, default="all", help="Specific gesture name, 'all', or 'missing'")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--status", action="store_true", help="Print dataset collection status report and exit")
    parser.add_argument("--missing", action="store_true", help="Capture only classes that are missing or under-sampled")
    args = parser.parse_args()

    if args.status:
        print_dataset_status(min_images=args.count)
        sys.exit(0)

    if args.missing or args.gesture.lower() == "missing":
        missing_list = get_missing_classes(min_images=args.count)
        if not missing_list:
            print(f"[+] All {len(GESTURE_CLASSES)} gesture classes already have at least {args.count} images!")
        else:
            missing_names = [name for name, _ in missing_list]
            print(f"[!] Found {len(missing_names)} classes needing more data: {missing_names}")
            capture_dataset_workflow(target_classes=missing_names, images_per_class=args.count, device_index=args.device)

    elif args.gesture.lower() == "all":
        capture_dataset_workflow(target_classes=GESTURE_CLASSES, images_per_class=args.count, device_index=args.device)

    else:
        # Match specific class
        matches = [g for g in GESTURE_CLASSES if g.lower() == args.gesture.lower()]
        if not matches:
            print(f"[!] Unknown gesture: {args.gesture}.")
            print(f"    Available classes: {GESTURE_CLASSES}")
        else:
            capture_gesture_images(matches[0], target_count=args.count, device_index=args.device)
            process_image_directory()

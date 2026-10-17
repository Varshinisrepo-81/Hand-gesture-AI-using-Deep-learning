"""
Webcam Camera Manager Module.
Provides robust camera initialization, frame capture, and resource cleanup with OpenCV.
Handles hardware disconnects, Windows DSHOW fallback, and camera index resolution.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


class CameraManager:
    """
    Manages OpenCV VideoCapture lifecycle with high-definition stream configuration
    and real-time computational image quality enhancements (CLAHE, sharpening, denoising).
    """

    # Standard quality presets
    PRESETS = {
        "720p HD (1280x720)": (1280, 720),
        "1080p Full HD (1920x1080)": (1920, 1080),
        "480p SD (640x480)": (640, 480)
    }

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = 30.0

        # Enhancement configurations
        self.enable_clahe = True
        self.enable_sharpening = True
        self.enable_denoising = False
        self.brightness = 0
        self.contrast = 1.0
        self._clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def list_available_cameras(max_tested: int = 4) -> list:
        """
        Tests camera indices from 0 up to max_tested and returns available indices.
        """
        available = []
        for idx in range(max_tested):
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
            opened = False
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available.append(idx)
                            opened = True
                        cap.release()
                        if opened:
                            break
                except Exception:
                    pass
        return available if available else [0]

    def set_enhancements(
        self,
        enable_clahe: bool = True,
        enable_sharpening: bool = True,
        enable_denoising: bool = False,
        brightness: int = 0,
        contrast: float = 1.0
    ) -> None:
        """Configures real-time image quality enhancement parameters."""
        self.enable_clahe = enable_clahe
        self.enable_sharpening = enable_sharpening
        self.enable_denoising = enable_denoising
        self.brightness = max(-100, min(100, int(brightness)))
        self.contrast = max(0.5, min(2.0, float(contrast)))

    def enhance_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Applies real-time image quality enhancements:
        1. CLAHE in LAB space for adaptive lighting & shadow recovery
        2. Denoising (optional) for low-light sensor grain removal
        3. Unsharp mask kernel for crisp hand contours and knuckle clarity
        4. Brightness & contrast scaling
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        enhanced = frame_bgr

        # 1. CLAHE Adaptive Lighting Equalization (LAB color space)
        if self.enable_clahe:
            try:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = self._clahe_processor.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        # 2. Subtle Gaussian/Bilateral Denoising (if enabled)
        if self.enable_denoising:
            try:
                # Fast bilateral filter preserves sharp hand edges while smoothing noise
                enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
            except Exception:
                pass

        # 3. Unsharp Masking / Edge Sharpening for crisp landmarks
        if self.enable_sharpening:
            try:
                # Smooth blur layer
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
                # Weighted difference gives a very natural crispness without halo artifacts
                enhanced = cv2.addWeighted(enhanced, 1.35, gaussian, -0.35, 0)
            except Exception:
                pass

        # 4. Brightness and Contrast Scaling
        if self.brightness != 0 or abs(self.contrast - 1.0) > 0.01:
            try:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=self.contrast, beta=self.brightness)
            except Exception:
                pass

        return enhanced

    def start(self, device_index: Optional[int] = None) -> Tuple[bool, str]:
        """
        Attempts to open webcam with hardware HD configuration (MJPG, 720p/1080p, buffer=1)
        and multiple backend fallbacks (DSHOW -> MSMF -> ANY).
        """
        if device_index is not None:
            self.device_index = device_index

        self.stop()  # Ensure previous instance released

        # Multi-backend fallback chain on Windows
        backends = []
        if os.name == "nt":
            backends = [
                ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
                ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
                ("Default (CAP_ANY)", cv2.CAP_ANY)
            ]
        else:
            backends = [("Default", cv2.CAP_ANY)]

        last_err = ""
        for backend_name, backend_flag in backends:
            try:
                self.cap = cv2.VideoCapture(self.device_index, backend_flag)
                if self.cap.isOpened():
                    # 1. Set MJPG fourcc codec to unlock high resolutions (720p/1080p) at 30+ FPS
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass

                    # 2. Request desired resolution & framerate
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                    # 3. Set buffer size to 1 to eliminate frame queue lag (instant real-time response)
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    # 4. Enable auto focus and auto exposure if supported by hardware
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    except Exception:
                        pass

                    # Warm up test read
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.is_running = True
                        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
                        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
                        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
                        return True, (
                            f"Camera opened successfully using {backend_name} on device {self.device_index} "
                            f"({self.actual_width}x{self.actual_height} @ {self.actual_fps:.0f} FPS)."
                        )
                    else:
                        self.cap.release()
                        last_err = f"Backend {backend_name} opened device {self.device_index} but failed to capture frames."
            except Exception as e:
                last_err = f"Backend {backend_name} encountered error: {e}"

        self.is_running = False
        available = self.list_available_cameras()
        return False, (
            f"Could not access camera at device index {self.device_index}. {last_err}\n"
            f"Detected available camera devices: {available}. "
            "Please check privacy settings, select another device index, or close other software using the camera."
        )

    def read_frame(self, apply_enhancements: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single BGR frame from the active camera with error guarding,
        mirror flip, and optional real-time image quality enhancements.
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

            # Flip horizontally for natural mirror-like user experience
            frame = cv2.flip(frame, 1)

            # Apply computer vision quality enhancements
            if apply_enhancements:
                frame = self.enhance_frame(frame)

            return True, frame
        except Exception:
            return False, None

    def stop(self) -> None:
        """Releases the camera device and cleans up resources safely."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def is_active(self) -> bool:
        """Returns True if the camera is currently running."""
        return self.is_running and self.cap is not None and self.cap.isOpened()

    def get_resolution_string(self) -> str:
        """Returns a formatted resolution string e.g. '1280x720 (HD)'."""
        w, h = self.actual_width, self.actual_height
        if w >= 1920 or h >= 1080:
            tag = "FHD"
        elif w >= 1280 or h >= 720:
            tag = "HD"
        else:
            tag = "SD"
        return f"{w}x{h} ({tag})"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

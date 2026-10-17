"""
Integration tests for Streamlit application components and end-to-end imports.
"""

import unittest
import numpy as np
from src.ui import GESTURE_ICONS, GESTURE_DESCRIPTIONS
from src.preprocessor import GESTURE_CLASSES, normalize_landmarks
from src.predictor import GesturePredictor
from src.camera import CameraManager
from database.db_manager import DatabaseManager


class TestAppIntegration(unittest.TestCase):

    def test_gesture_classes_consistency(self):
        """Verifies all gesture classes have icons and descriptions."""
        self.assertEqual(len(GESTURE_CLASSES), 32)
        for g in GESTURE_CLASSES:
            self.assertIn(g, GESTURE_ICONS)
            self.assertIn(g, GESTURE_DESCRIPTIONS)

    def test_camera_manager_initialization(self):
        """Checks camera manager default instantiation, resolution presets, and enhancements."""
        cam = CameraManager(device_index=0, width=1280, height=720)
        self.assertEqual(cam.device_index, 0)
        self.assertEqual(cam.width, 1280)
        self.assertEqual(cam.height, 720)
        self.assertFalse(cam.is_active())
        self.assertIn("HD", cam.get_resolution_string())

        # Test frame enhancement pipeline on synthetic frame
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        cam.set_enhancements(enable_clahe=True, enable_sharpening=True, enable_denoising=True)
        enhanced = cam.enhance_frame(dummy_frame)
        self.assertEqual(enhanced.shape, dummy_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        cam.stop()
        self.assertFalse(cam.is_active())

    def test_predictor_and_database_interaction(self):
        """Tests predicting a sample and logging it into SQLite."""
        db = DatabaseManager()
        predictor = GesturePredictor()

        # Mock random normalized hand coordinates
        dummy_landmarks = np.zeros((21, 3), dtype=np.float32)
        dummy_landmarks[4] = [0.1, -0.2, 0.0]
        dummy_landmarks[8] = [0.0, -0.3, 0.0]

        res = predictor.predict(dummy_landmarks, confidence_threshold=0.50)
        self.assertIn("gesture", res)
        self.assertIn("confidence", res)

        # Log prediction to DB
        row_id = db.log_prediction(
            gesture=res["raw_gesture"],
            confidence=res["confidence"],
            landmarks_count=21,
            notes="Integration test record"
        )
        self.assertGreater(row_id, 0)

        # Verify entry is found in DB
        df = db.get_predictions_df(limit=5)
        self.assertGreater(len(df), 0)
        latest = df.iloc[0]
        self.assertEqual(latest["notes"], "Integration test record")


if __name__ == "__main__":
    unittest.main()

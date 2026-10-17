"""
Unit Tests for Predictor Module.
"""

import unittest
import numpy as np
from src.predictor import GesturePredictor
from src.preprocessor import GESTURE_CLASSES, NUM_LANDMARKS
from data.generate_dataset import generate_single_gesture_skeleton


class TestGesturePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = GesturePredictor()

    def test_model_is_loaded(self):
        self.assertTrue(self.predictor.is_model_loaded(), "Model should be loaded from disk.")

    def test_prediction_output_format(self):
        # Generate Open Palm skeleton
        palm_skeleton = generate_single_gesture_skeleton(0)
        res = self.predictor.predict(palm_skeleton, confidence_threshold=0.50)

        self.assertIn("gesture", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("is_low_confidence", res)
        self.assertIsInstance(res["confidence"], float)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        # Check all 5 classes are present in probabilities dict
        for g in GESTURE_CLASSES:
            self.assertIn(g, res["probabilities"])

    def test_prediction_gestures_classification(self):
        # Test each of the 5 gestures
        for idx, expected_gesture in enumerate(GESTURE_CLASSES):
            skeleton = generate_single_gesture_skeleton(idx)
            res = self.predictor.predict(skeleton, confidence_threshold=0.50)
            self.assertEqual(res["raw_gesture"], expected_gesture, f"Expected {expected_gesture}, got {res['raw_gesture']}")

    def test_low_confidence_flag(self):
        skeleton = generate_single_gesture_skeleton(1)  # Fist
        # Set threshold of 1.01 to guarantee triggering the low-confidence branch
        res = self.predictor.predict(skeleton, confidence_threshold=1.01)
        self.assertTrue(res["is_low_confidence"])
        self.assertEqual(res["gesture"], "Uncertain / Low Confidence")
        # Raw gesture is still preserved
        self.assertEqual(res["raw_gesture"], "Fist")


if __name__ == "__main__":
    unittest.main()

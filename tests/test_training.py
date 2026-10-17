"""
Unit Tests for Training and Evaluation Pipeline.
"""

import unittest
import os
import numpy as np
from src.model import build_gesture_model
from src.evaluate import evaluate_model_on_cache, get_latest_metrics
from src.preprocessor import GESTURE_CLASSES, FEATURE_DIM


class TestTrainingAndEvaluation(unittest.TestCase):

    def test_model_architecture(self):
        model = build_gesture_model(input_dim=FEATURE_DIM, num_classes=len(GESTURE_CLASSES))
        self.assertEqual(model.input_shape, (None, 63))
        self.assertEqual(model.output_shape, (None, len(GESTURE_CLASSES)))

    def test_evaluation_on_cache(self):
        metrics = evaluate_model_on_cache()
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Check values are in valid ranges
        self.assertGreaterEqual(metrics["accuracy"], 0.85)
        self.assertGreaterEqual(metrics["f1_macro"], 0.85)
        self.assertEqual(len(metrics["confusion_matrix"]), len(GESTURE_CLASSES))

    def test_latest_metrics_file(self):
        saved = get_latest_metrics()
        self.assertIsNotNone(saved)
        self.assertEqual(saved["classes"], GESTURE_CLASSES)


if __name__ == "__main__":
    unittest.main()

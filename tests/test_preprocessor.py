"""
Unit Tests for Preprocessor Module.
"""

import unittest
import numpy as np
from src.preprocessor import (
    normalize_landmarks,
    batch_normalize,
    FEATURE_DIM,
    NUM_LANDMARKS,
    GESTURE_CLASSES
)


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        # Create a mock 21-joint landmark array (21, 3)
        self.mock_landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        # Set distinct points
        for i in range(NUM_LANDMARKS):
            self.mock_landmarks[i] = [i * 0.05, -i * 0.03, i * 0.01]

    def test_output_shape_and_dtype(self):
        norm = normalize_landmarks(self.mock_landmarks)
        self.assertEqual(norm.shape, (FEATURE_DIM,))
        self.assertEqual(norm.dtype, np.float32)

    def test_wrist_at_origin(self):
        norm = normalize_landmarks(self.mock_landmarks)
        # Landmark 0 should be normalized to (0, 0, 0)
        self.assertAlmostEqual(float(norm[0]), 0.0, places=5)
        self.assertAlmostEqual(float(norm[1]), 0.0, places=5)
        self.assertAlmostEqual(float(norm[2]), 0.0, places=5)

    def test_scale_invariance(self):
        # Scale all coordinates by 3.5x
        scaled_landmarks = self.mock_landmarks * 3.5
        norm_orig = normalize_landmarks(self.mock_landmarks)
        norm_scaled = normalize_landmarks(scaled_landmarks)

        # Normalized values should be identical regardless of scale!
        np.testing.assert_allclose(norm_orig, norm_scaled, atol=1e-5)

    def test_translation_invariance(self):
        # Shift entire hand in 3D space
        shifted_landmarks = self.mock_landmarks + np.array([10.5, -4.2, 8.0], dtype=np.float32)
        norm_orig = normalize_landmarks(self.mock_landmarks)
        norm_shifted = normalize_landmarks(shifted_landmarks)

        # Normalized values should be identical regardless of spatial position!
        np.testing.assert_allclose(norm_orig, norm_shifted, atol=1e-5)

    def test_invalid_shape_raises_error(self):
        invalid_shape = np.zeros((10, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize_landmarks(invalid_shape)

    def test_batch_normalize(self):
        batch = np.array([self.mock_landmarks, self.mock_landmarks * 2.0])
        norm_batch = batch_normalize(batch)
        self.assertEqual(norm_batch.shape, (2, FEATURE_DIM))
        np.testing.assert_allclose(norm_batch[0], norm_batch[1], atol=1e-5)


if __name__ == "__main__":
    unittest.main()

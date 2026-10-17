"""
Unit Tests for Database Manager Module.
"""

import unittest
import os
import tempfile
import pandas as pd
from database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_gestures.db")
        self.db = DatabaseManager(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_and_fetch_prediction(self):
        row_id = self.db.log_prediction(
            gesture="Open Palm",
            confidence=0.9850,
            landmarks_count=21,
            notes="Test prediction"
        )
        self.assertGreater(row_id, 0)

        df = self.db.get_predictions_df(limit=10)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["gesture"], "Open Palm")
        self.assertAlmostEqual(float(df.iloc[0]["confidence"]), 0.9850, places=3)
        self.assertEqual(df.iloc[0]["notes"], "Test prediction")

    def test_filter_predictions(self):
        self.db.log_prediction("Fist", 0.95)
        self.db.log_prediction("Peace Sign", 0.88)
        self.db.log_prediction("Fist", 0.70)

        df_fist = self.db.get_predictions_df(gesture_filter="Fist")
        self.assertEqual(len(df_fist), 2)

        df_high_conf = self.db.get_predictions_df(min_confidence=0.80)
        self.assertEqual(len(df_high_conf), 2)

    def test_statistics(self):
        self.db.log_prediction("Thumbs Up", 0.90)
        self.db.log_prediction("Thumbs Up", 0.96)
        self.db.log_prediction("Peace Sign", 0.80)

        stats = self.db.get_prediction_statistics()
        self.assertEqual(stats["total_predictions"], 3)
        self.assertEqual(stats["most_frequent_gesture"], "Thumbs Up")
        self.assertGreater(stats["average_confidence"], 0.85)

    def test_csv_export(self):
        self.db.log_prediction("Pointing Finger", 0.92)
        csv_bytes = self.db.export_predictions_csv_bytes()
        self.assertIsInstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        self.assertIn("Pointing Finger", csv_str)
        self.assertIn("confidence", csv_str)

    def test_clear_predictions(self):
        self.db.log_prediction("Fist", 0.99)
        self.assertEqual(len(self.db.get_predictions_df()), 1)
        self.db.clear_predictions()
        self.assertEqual(len(self.db.get_predictions_df()), 0)

    def test_training_session_log(self):
        row_id = self.db.log_training_session(
            epochs=25,
            train_samples=1750,
            test_samples=375,
            test_accuracy=0.992,
            test_loss=0.035,
            model_path="models/gesture_model.keras"
        )
        self.assertGreater(row_id, 0)
        df_train = self.db.get_training_sessions_df()
        self.assertEqual(len(df_train), 1)
        self.assertEqual(int(df_train.iloc[0]["epochs"]), 25)
        self.assertAlmostEqual(float(df_train.iloc[0]["test_accuracy"]), 0.992, places=3)


if __name__ == "__main__":
    unittest.main()

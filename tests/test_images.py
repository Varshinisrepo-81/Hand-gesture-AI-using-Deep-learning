"""
Unit Tests for Image Dataset Processing and 100 Sample Image Generation.
"""

import os
import unittest
from data.process_images import IMAGES_DIR, ensure_image_directories
from src.preprocessor import GESTURE_CLASSES


class TestImageDataset(unittest.TestCase):

    def test_image_directories_exist(self):
        """Verifies subdirectories exist for each gesture class."""
        dirs = ensure_image_directories(IMAGES_DIR)
        for g in GESTURE_CLASSES:
            self.assertIn(g, dirs)
            self.assertTrue(os.path.isdir(dirs[g]))

    def test_sample_images_present(self):
        """Verifies that at least 100 sample images exist on disk."""
        dirs = ensure_image_directories(IMAGES_DIR)
        total_found = 0
        for g, path in dirs.items():
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                total_found += len(files)

        self.assertGreaterEqual(total_found, 100, f"Expected at least 100 images, found {total_found}")


if __name__ == "__main__":
    unittest.main()

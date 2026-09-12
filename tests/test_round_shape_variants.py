import unittest

import cv2
import numpy as np

from app.modeling.shape_recognizer import ShapeRecognizer


class RoundShapeVariantTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = ShapeRecognizer(debug=False)

    def test_perfect_small_and_large_circles_remain_circles(self):
        for radius in (32, 110):
            with self.subTest(radius=radius):
                image = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.circle(image, (150, 150), radius, (255, 255, 255), 7, cv2.LINE_AA)
                result = self.recognizer.recognize(image)
                self.assertEqual(result.shape.value, "circle")
                self.assertGreaterEqual(result.confidence, 0.78)

    def test_irregular_closed_circle_uses_radius_and_contour_evidence(self):
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radii = 90 * (1 + 0.07 * np.sin(5 * angles))
        points = np.column_stack((150 + radii * np.cos(angles), 150 + radii * np.sin(angles))).astype(np.int32)
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.polylines(image, [points], True, (255, 255, 255), 7, cv2.LINE_AA)
        result = self.recognizer.recognize(image)
        self.assertEqual(result.shape.value, "circle")
        self.assertGreater(result.features.radius_consistency, 0.8)


if __name__ == "__main__":
    unittest.main()

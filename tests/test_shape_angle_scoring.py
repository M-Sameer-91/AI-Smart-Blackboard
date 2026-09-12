import unittest

import cv2
import numpy as np

from app.modeling.shape_recognizer import ShapeRecognizer


def outlined(points):
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], True, (255, 255, 255), 10, cv2.LINE_AA)
    return image


class ShapeAngleScoringTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = ShapeRecognizer(debug=False)

    def test_rotated_imperfect_square_uses_angle_evidence(self):
        result = self.recognizer.recognize(outlined([(250, 70), (425, 245), (245, 430), (65, 250)]))
        self.assertEqual(result.shape.value, "square")
        self.assertGreater(result.features.angle_score, 0.6)
        self.assertEqual(len(result.features.internal_angles), 4)

    def test_regular_polygon_records_expected_angle_evidence(self):
        points = [(250 + 150 * np.cos(-np.pi / 2 + i * 2 * np.pi / 6),
                   250 + 150 * np.sin(-np.pi / 2 + i * 2 * np.pi / 6)) for i in range(6)]
        result = self.recognizer.recognize(outlined(points))
        self.assertEqual(result.shape.value, "hexagon")
        self.assertGreater(result.features.angle_score, 0.6)


if __name__ == "__main__":
    unittest.main()

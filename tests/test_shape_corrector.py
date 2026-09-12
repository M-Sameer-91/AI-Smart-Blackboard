"""Geometry-level tests for central auto-correction."""

import math
import unittest

from app.modeling.shape_corrector import ShapeCorrector
from app.modeling.shape_recognizer import ShapeFeatures, ShapeType


class ShapeCorrectorTests(unittest.TestCase):
    def test_circle_uses_contour_fit_not_bounding_box_radius(self):
        # Bounding box deliberately differs from the fitted 40px circle.
        contour = [(140 + 40 * math.cos(angle), 90 + 40 * math.sin(angle)) for angle in [i * math.pi / 12 for i in range(24)]]
        corrected = ShapeCorrector().correct(ShapeType.CIRCLE, ShapeFeatures(bounding_rect=(20, 20, 240, 150)), contour)
        radii = [math.dist(point, corrected.center) for point in corrected.points[:-1]]
        self.assertAlmostEqual(corrected.center[0], 140, delta=1)
        self.assertAlmostEqual(corrected.center[1], 90, delta=1)
        self.assertAlmostEqual(sum(radii) / len(radii), 40, delta=1)
        self.assertLess(max(radii) - min(radii), 0.001)

    def test_square_has_equal_sides_and_star_is_closed(self):
        corrector = ShapeCorrector()
        square = corrector.correct(ShapeType.SQUARE, ShapeFeatures(bounding_rect=(20, 30, 100, 80)))
        sides = [math.dist(square.points[i], square.points[i + 1]) for i in range(4)]
        self.assertLess(max(sides) - min(sides), 0.001)
        star = corrector.correct(ShapeType.STAR, ShapeFeatures(bounding_rect=(20, 30, 100, 80)))
        self.assertEqual(len(star.points), 11)
        self.assertEqual(star.points[0], star.points[-1])


if __name__ == "__main__":
    unittest.main()

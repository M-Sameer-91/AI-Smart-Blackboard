"""Regression tests for generalized 2D polygon recognition and 3D extrusion."""

import unittest

import cv2
import numpy as np

from app.modeling.geometry_generator import GeometryGenerator
from app.modeling.shape_recognizer import RecognitionResult, ShapeFeatures, ShapeRecognizer, ShapeType


def outlined(points, size=500):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.polylines(image, [np.asarray(points, dtype=np.int32)], True, (255, 255, 255), 10, cv2.LINE_AA)
    return image


def regular_polygon(sides, radius=150, center=(250, 250)):
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, sides, endpoint=False)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


class ExtendedShapeTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = ShapeRecognizer(debug=False)

    def test_regular_polygons_are_recognized(self):
        for name, sides in (("pentagon", 5), ("hexagon", 6), ("octagon", 8)):
            with self.subTest(shape=name):
                result = self.recognizer.recognize(outlined(regular_polygon(sides)))
                self.assertEqual(result.shape.value, name)
                self.assertGreaterEqual(result.confidence, 0.70)

    def test_star_and_trapezoid_are_recognized(self):
        shapes = {
            "trapezoid": [(100, 100), (400, 100), (330, 400), (170, 400)],
            "star": [(250, 80), (285, 190), (400, 190), (307, 255), (345, 380), (250, 305), (155, 380), (193, 255), (100, 190), (215, 190)],
        }
        for name, points in shapes.items():
            with self.subTest(shape=name):
                result = self.recognizer.recognize(outlined(points))
                self.assertEqual(result.shape.value, name)
                self.assertGreaterEqual(result.confidence, 0.70)

    def test_primitives_and_generalized_extrusions_are_valid(self):
        generator = GeometryGenerator()
        meshes = [
            generator.create_cube(20),
            generator.create_cylinder(10, 20),
            generator.create_triangular_prism(20, 18, 12),
            generator.create_polygon_prism(regular_polygon(5, 10, (0, 0)), 12),
            generator.create_polygon_prism(regular_polygon(6, 10, (0, 0)), 12),
            generator.create_polygon_prism(generator._template_polygon("star", 20, 20), 12),
        ]
        for mesh in meshes:
            self.assertTrue(generator.validate_mesh(mesh))
            self.assertGreater(len(mesh.vertices), 0)
            self.assertGreater(len(mesh.faces), 0)
            self.assertGreater(mesh.volume, 0)

    def test_extended_shape_results_generate_watertight_prisms(self):
        generator = GeometryGenerator()
        for shape in (ShapeType.PENTAGON, ShapeType.HEXAGON, ShapeType.OCTAGON, ShapeType.STAR, ShapeType.TRAPEZOID):
            with self.subTest(shape=shape.value):
                result = generator.generate(RecognitionResult(shape, 0.70, ShapeFeatures(width=120, height=100, area=9000)))
                self.assertTrue(result.success, result.error)
                self.assertTrue(result.is_watertight)

    def test_confidence_boundary_and_unknown_are_safe(self):
        generator = GeometryGenerator()
        features = ShapeFeatures(width=100, height=100, area=7000)
        self.assertFalse(generator.generate(RecognitionResult(ShapeType.PENTAGON, 0.69, features)).success)
        self.assertTrue(generator.generate(RecognitionResult(ShapeType.PENTAGON, 0.70, features)).success)
        unknown = generator.generate(RecognitionResult(ShapeType.UNKNOWN, 0.90, features))
        self.assertFalse(unknown.success)
        self.assertIn("Unsupported shape", unknown.error)


if __name__ == "__main__":
    unittest.main()

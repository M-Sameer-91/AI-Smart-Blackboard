"""End-to-end regression tests for the blackboard Shape -> 3D workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from app.modeling.geometry_generator import GeometryGenerator
from app.modeling.model_pipeline import ModelPipeline
from app.modeling.shape_recognizer import RecognitionResult, ShapeFeatures, ShapeRecognizer, ShapeType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "circle": "clean_circle.png",
    "square": "clean_square.png",
    "rectangle": "clean_rectangle.png",
    "triangle": "clean_triangle.png",
    "ellipse": "clean_ellipse.png",
}


class ShapeTo3DTests(unittest.TestCase):
    def test_clean_shapes_recognize_and_export_watertight_stl(self):
        with TemporaryDirectory() as directory:
            pipeline = ModelPipeline(output_dir=directory)
            for expected, filename in SHAPES.items():
                with self.subTest(shape=expected):
                    result = pipeline.process(PROJECT_ROOT / "test_shapes" / filename)
                    self.assertTrue(result["success"], result["error"])
                    self.assertEqual(result["recognition"]["shape"], expected)
                    self.assertGreaterEqual(result["recognition"]["confidence"], 0.70)
                    self.assertTrue(result["geometry"]["watertight"])
                    output = Path(result["stl"]["path"])
                    self.assertTrue(output.is_file())
                    self.assertGreater(output.stat().st_size, 0)

    def test_recognition_dataclass_flows_into_geometry_generator(self):
        result = RecognitionResult(
            shape=ShapeType.CIRCLE,
            confidence=0.90,
            features=ShapeFeatures(width=100, height=100, area=7850, perimeter=314),
        )
        geometry = GeometryGenerator().generate(result)
        self.assertTrue(geometry.success, geometry.error)
        self.assertTrue(geometry.is_watertight)

    def test_low_confidence_is_rejected_before_export(self):
        with TemporaryDirectory() as directory:
            pipeline = ModelPipeline(output_dir=directory)
            pipeline.shape_recognizer.recognize = lambda _: RecognitionResult(
                ShapeType.CIRCLE, 0.69, ShapeFeatures(width=100, height=100)
            )
            result = pipeline.process(PROJECT_ROOT / "test_shapes" / "clean_circle.png")
            self.assertFalse(result["success"])
            self.assertEqual(result["stage"], "confidence")
            self.assertIn("below threshold", result["error"])

    def test_empty_drawing_returns_recognition_error(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "empty.png"
            cv2.imwrite(str(image_path), np.zeros((200, 200, 3), dtype=np.uint8))
            result = ModelPipeline(output_dir=directory).process(image_path)
            self.assertFalse(result["success"])
            self.assertEqual(result["stage"], "recognition")
            self.assertIn("No contours found", result["error"])


if __name__ == "__main__":
    unittest.main()

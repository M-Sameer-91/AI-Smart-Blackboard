"""Contract tests for the single grouped-recognition entry point."""

import unittest

import cv2
import numpy as np

from app.recognition.orchestrator import RecognitionOrchestrator


class _OCR:
    def __init__(self, text: str, confidence: float = 0.95):
        self.result = {"text": text, "confidence": confidence}

    def extract_text_with_confidence(self, _path):
        return self.result


class RecognitionOrchestratorTests(unittest.TestCase):
    def test_equation_is_classified_and_parsed_as_math(self):
        result = RecognitionOrchestrator(_OCR("2x + 5 = 15")).analyze(
            np.zeros((120, 400, 3), dtype=np.uint8)
        )
        self.assertEqual(result.kind, "math")
        self.assertEqual(result.text, "2x + 5 = 15")
        self.assertIsNotNone(result.math)
        self.assertTrue(result.math.parsed)

    def test_circle_stays_shape_and_exposes_geometry(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(image, (150, 150), 90, (255, 255, 255), 9, cv2.LINE_AA)
        result = RecognitionOrchestrator(_OCR("No text recognized", 0.0)).analyze(image)
        self.assertEqual(result.kind, "shape")
        self.assertEqual(result.label, "circle")
        board_object = result.to_board_object((2, 3), corrected=True)
        self.assertEqual(board_object.kind, "shape")
        self.assertTrue(board_object.corrected)
        self.assertEqual(board_object.source_stroke_range, (2, 3))

    def test_single_letter_remains_character(self):
        result = RecognitionOrchestrator(_OCR("y")).analyze(np.zeros((120, 120, 3), dtype=np.uint8))
        self.assertEqual(result.kind, "character")
        self.assertEqual(result.label, "y")


if __name__ == "__main__":
    unittest.main()

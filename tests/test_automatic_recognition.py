"""Focused coverage for automatic shape-versus-text arbitration."""

import unittest

import cv2
import numpy as np

from app.recognition.automatic import AutomaticRecognitionEngine


class _OCR:
    def __init__(self, text="No text recognized", confidence=0.0):
        self.result = {"text": text, "confidence": confidence}

    def extract_text_with_confidence(self, _path):
        return self.result


def circle_image():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(image, (150, 150), 90, (255, 255, 255), 9, cv2.LINE_AA)
    return image


class AutomaticRecognitionTests(unittest.TestCase):
    def test_confident_geometry_beats_empty_ocr(self):
        result = AutomaticRecognitionEngine(_OCR()).analyze(circle_image())
        self.assertEqual(result.kind, "shape")
        self.assertEqual(result.detected, "circle")
        self.assertGreaterEqual(result.confidence, 0.70)
        self.assertIsNotNone(result.shape_result)

    def test_single_letter_is_reported_as_character_when_shape_is_not_confident(self):
        image = np.zeros((220, 180, 3), dtype=np.uint8)
        cv2.putText(image, "A", (35, 170), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8, cv2.LINE_AA)
        result = AutomaticRecognitionEngine(_OCR("A", 0.95)).analyze(image)
        self.assertEqual(result.kind, "character")
        self.assertEqual(result.detected, "A")

    def test_multi_character_ocr_is_reported_as_text(self):
        result = AutomaticRecognitionEngine(_OCR("Hello", 0.85)).analyze(np.zeros((120, 360, 3), dtype=np.uint8))
        self.assertEqual(result.kind, "text")
        self.assertEqual(result.text, "Hello")

    def test_round_character_wins_when_shape_correction_is_off(self):
        result = AutomaticRecognitionEngine(_OCR("o", 0.84)).analyze(circle_image(), shape_correction_enabled=False)
        self.assertEqual(result.kind, "character")
        self.assertEqual(result.detected, "o")

    def test_strong_circle_remains_shape_when_correction_is_on(self):
        result = AutomaticRecognitionEngine(_OCR("o", 0.84)).analyze(circle_image(), shape_correction_enabled=True)
        self.assertEqual(result.kind, "shape")


if __name__ == "__main__":
    unittest.main()

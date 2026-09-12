import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.recognition.font_evaluation import _character_accuracy, OCRFontEvaluator


class _OCR:
    def is_model_loaded(self):
        return True

    def extract_text_with_confidence(self, _path):
        return {"text": "Hello", "confidence": 0.9}


class FontEvaluationTests(unittest.TestCase):
    def test_character_accuracy_normalises_case_and_spacing(self):
        self.assertEqual(_character_accuracy("Hello World", "hello   world"), 1.0)
        self.assertLess(_character_accuracy("Circle", "o"), 0.5)

    def test_unavailable_font_is_reported_explicitly(self):
        with TemporaryDirectory() as directory:
            result = OCRFontEvaluator(Path(directory), Path(directory) / "out", _OCR()).evaluate_font("Caveat")
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

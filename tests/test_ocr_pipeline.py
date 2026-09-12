"""Regression coverage for canvas capture and the OCR input contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.recognition.ocr import OCRProcessor
from app.recognition.preprocessor import ImagePreprocessor
from app.recognition.recognizer import Recognizer


class _Stroke:
    def __init__(self, points, color="white", width=4):
        self.points = points
        self.color = color
        self.width = width


class _Canvas:
    def get_canvas_size(self):
        return (160, 100)

    def get_stroke_history(self):
        return [_Stroke([(15, 50), (145, 50)])]

    def get_stroke_count(self):
        return 1


class OCRPipelineTests(unittest.TestCase):
    def _write_blackboard_text(self, directory: str, text: str, font_size: int = 88) -> Path:
        image = Image.new("RGB", (900, 280), "black")
        draw = ImageDraw.Draw(image)
        font_path = Path(r"C:\Windows\Fonts\segoepr.ttf")
        font = ImageFont.truetype(str(font_path), font_size) if font_path.exists() else ImageFont.load_default()
        draw.text((45, 70), text, fill="white", font=font)
        path = Path(directory) / "blackboard_text.png"
        image.save(path)
        return path

    def test_preprocessor_preserves_white_blackboard_ink_as_black_on_white(self):
        with TemporaryDirectory() as directory:
            path = self._write_blackboard_text(directory, "Hello")
            page = np.asarray(ImagePreprocessor().preprocess(path).convert("L"))
            self.assertLess(np.count_nonzero(page < 127), page.size * 0.30)
            self.assertGreater(np.count_nonzero(page < 127), 50)
            self.assertGreater(np.count_nonzero(page > 127), page.size * 0.60)

    def test_canvas_capture_renders_strokes_without_screen_grab_or_transparency(self):
        with TemporaryDirectory() as directory:
            capture = Recognizer(_Canvas())
            capture._output_dir = Path(directory)
            path = capture.capture_canvas()
            image = np.asarray(Image.open(path).convert("RGB"))
            self.assertEqual(image.shape, (100, 160, 3))
            self.assertGreater(np.count_nonzero(image), 100)

    def test_empty_canvas_image_has_useful_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.png"
            cv2.imwrite(str(path), np.zeros((240, 400, 3), dtype=np.uint8))
            result = OCRProcessor().extract_text_with_confidence(path)
            self.assertIn("Empty drawing", result["text"])
            self.assertEqual(result["confidence"], 0.0)

    def test_handwriting_font_word_sentence_and_character_reach_tesseract(self):
        processor = OCRProcessor()
        if not processor.is_model_loaded():
            self.skipTest("Tesseract is not installed")
        with TemporaryDirectory() as directory:
            for text in ("Hello", "Hello World", "A"):
                with self.subTest(text=text):
                    result = processor.extract_text_with_confidence(self._write_blackboard_text(directory, text))
                    self.assertNotIn("[OCR Error]", result["text"])
                    self.assertNotEqual(result["text"], "No text recognized")
                    self.assertGreater(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Controlled OCR evaluation for handwriting-style fonts.

This is development tooling only.  It renders known labels to canvas-like
images and sends them through the production OCRProcessor; it never changes
the runtime recognizer or claims that fonts represent human handwriting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from app.recognition.ocr import OCRProcessor


FONT_NAMES = ("Caveat", "IndieFlower", "DancingScript", "Handlee")
CHARACTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
AMBIGUOUS = ("o", "O", "c", "e", "a", "x", "y", "l", "I", "1", "0")
WORDS = ("Hello", "World", "Smart", "Blackboard", "Computer", "Science", "Mathematics", "Circle", "Square", "Triangle")
PROJECT_TEXT = ("AI Smart Blackboard", "Smart Blackboard", "Shape Correct", "Automatic Recognition")


@dataclass(frozen=True)
class FontEvaluationItem:
    expected: str
    detected: str
    confidence: float
    passed: bool
    character_accuracy: float
    category: str


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _character_accuracy(expected: str, detected: str) -> float:
    """Levenshtein-derived accuracy without an additional package."""
    expected, detected = _normalise(expected), _normalise(detected)
    if not expected:
        return 1.0 if not detected else 0.0
    previous = list(range(len(detected) + 1))
    for left_index, left in enumerate(expected, 1):
        current = [left_index]
        for right_index, right in enumerate(detected, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1,
                               previous[right_index - 1] + (left != right)))
        previous = current
    return max(0.0, 1.0 - previous[-1] / len(expected))


class OCRFontEvaluator:
    """Render controlled samples and write JSON/Markdown reports."""

    def __init__(self, font_dir: Path | str = "app/data/training/ocr_font_tests/fonts",
                 output_dir: Path | str = "app/data/training/ocr_font_tests/reports",
                 ocr_processor: Any | None = None) -> None:
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)
        self.ocr = ocr_processor or OCRProcessor(save_processed_images=False)

    def samples(self) -> Iterable[tuple[str, str]]:
        yield from (("character", value) for value in CHARACTERS)
        yield from (("ambiguous", value) for value in AMBIGUOUS)
        yield from (("word", value) for value in WORDS)
        yield from (("project_text", value) for value in PROJECT_TEXT)

    def _font_path(self, name: str) -> Path | None:
        candidates = list(self.font_dir.glob(f"{name}*.ttf")) + list(self.font_dir.glob(f"{name}*.otf"))
        return candidates[0] if candidates else None

    @staticmethod
    def _render(text: str, font_path: Path, destination: Path) -> None:
        font = ImageFont.truetype(str(font_path), 92)
        probe = Image.new("RGB", (1, 1))
        bounds = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
        image = Image.new("RGB", (max(520, bounds[2] - bounds[0] + 100), max(220, bounds[3] - bounds[1] + 120)), "black")
        ImageDraw.Draw(image).text((50, 45), text, fill="white", font=font)
        image.save(destination)

    def evaluate_font(self, name: str) -> dict[str, Any]:
        font_path = self._font_path(name)
        if font_path is None:
            return {"font": name, "status": "unavailable", "items": [], "accuracy": 0.0, "mean_confidence": 0.0}
        if not getattr(self.ocr, "is_model_loaded", lambda: True)():
            return {"font": name, "status": "ocr_unavailable", "items": [], "accuracy": 0.0, "mean_confidence": 0.0}
        sample_dir = self.output_dir / "samples" / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        items: list[FontEvaluationItem] = []
        for index, (category, expected) in enumerate(self.samples()):
            path = sample_dir / f"{index:03d}_{category}.png"
            self._render(expected, font_path, path)
            result = self.ocr.extract_text_with_confidence(path)
            detected = str(result.get("text", ""))
            accuracy = _character_accuracy(expected, detected)
            items.append(FontEvaluationItem(expected, detected, float(result.get("confidence", 0.0)),
                                            _normalise(expected) == _normalise(detected), accuracy, category))
        return {
            "font": name, "status": "complete", "items": [asdict(item) for item in items],
            "accuracy": sum(item.character_accuracy for item in items) / len(items),
            "mean_confidence": sum(item.confidence for item in items) / len(items),
        }

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"purpose": "controlled font evaluation; not evidence of general human-handwriting accuracy",
                  "fonts": [self.evaluate_font(name) for name in FONT_NAMES]}
        (self.output_dir / "font_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = ["# OCR font evaluation", "", report["purpose"], ""]
        for font in report["fonts"]:
            lines.extend([f"## {font['font']}", "", f"Status: {font['status']}",
                          f"Character accuracy: {font['accuracy']:.1%}", f"Mean confidence: {font['mean_confidence']:.2f}", ""])
            for item in font["items"]:
                lines.append(f"- [{ 'PASS' if item['passed'] else 'FAIL' }] {item['category']}: expected `{item['expected']}`, detected `{item['detected']}`, confidence {item['confidence']:.2f}, accuracy {item['character_accuracy']:.1%}")
        (self.output_dir / "font_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
        return report


if __name__ == "__main__":
    OCRFontEvaluator().run()

"""One local entry point for grouped shape, OCR, and mathematics recognition."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from app.config import RECOGNITION_SETTINGS
from app.modeling.board_objects import BoardObject
from app.modeling.shape_recognizer import RecognitionResult
from app.recognition.automatic import AutomaticRecognitionEngine
from app.recognition.classifier import ContentClassifier, ContentCategory
from app.recognition.math_engine import MathEngine, MathRecognition


RecognitionKind = Literal["shape", "character", "text", "math", "unknown"]


@dataclass(frozen=True)
class UnifiedRecognitionResult:
    kind: RecognitionKind
    label: str
    confidence: float
    model: str
    shape_result: Optional[RecognitionResult] = None
    text: str = ""
    math: Optional[MathRecognition] = None

    @property
    def detected(self) -> str:
        """Compatibility name used by the existing result panel."""
        return self.label

    def to_board_object(self, stroke_range: tuple[int, int], corrected: bool = False) -> BoardObject:
        bounding_box = None
        geometry: Any = None
        if self.shape_result is not None:
            bounding_box = self.shape_result.features.bounding_rect
            geometry = self.shape_result.features
        return BoardObject(self.kind, self.label, self.confidence, bounding_box, geometry,
                           stroke_range, corrected, {"recognizer": self.model})


class RecognitionOrchestrator:
    """Arbitrate the existing local recognizers and add deterministic math parsing."""

    def __init__(self, ocr_processor: Any, classifier: Optional[ContentClassifier] = None,
                 correction_threshold: float = RECOGNITION_SETTINGS.shape_correction_threshold,
                 debug: bool = False) -> None:
        self.automatic_engine = AutomaticRecognitionEngine(
            ocr_processor, correction_threshold=correction_threshold, debug=debug
        )
        self.classifier = classifier or ContentClassifier()
        self.math_engine = MathEngine()

    def analyze(self, image: np.ndarray, shape_correction_enabled: bool = True) -> UnifiedRecognitionResult:
        automatic = self.automatic_engine.analyze(image, shape_correction_enabled=shape_correction_enabled)
        if automatic.kind in {"character", "text"}:
            classification = self.classifier.classify(automatic.text)
            if classification["category"] == ContentCategory.MATHEMATICS.value:
                math = self.math_engine.analyze(automatic.text, automatic.confidence)
                return UnifiedRecognitionResult("math", automatic.detected, automatic.confidence,
                                                f"{automatic.model} + MathEngine", text=automatic.text, math=math)
        return UnifiedRecognitionResult(automatic.kind, automatic.detected, automatic.confidence,
                                        automatic.model, automatic.shape_result, automatic.text)

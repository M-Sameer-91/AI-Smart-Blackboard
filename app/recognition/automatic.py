"""Automatic, grouped drawing recognition built on the existing OCR and shape modules."""

from dataclasses import dataclass
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Optional

import numpy as np

from app.config import RECOGNITION_SETTINGS
from app.modeling.shape_recognizer import RecognitionResult, ShapeRecognizer, ShapeType

DEFAULT_SHAPE_CORRECTION_THRESHOLD = RECOGNITION_SETTINGS.shape_correction_threshold


@dataclass(frozen=True)
class AutomaticRecognitionResult:
    """UI-safe result for the shape/text arbitration stage."""
    kind: Literal["shape", "character", "text", "unknown"]
    detected: str
    confidence: float
    model: str
    shape_result: Optional[RecognitionResult] = None
    text: str = ""


class AutomaticRecognitionEngine:
    """Choose between geometry and OCR without forcing drawings through OCR."""

    def __init__(self, ocr_processor: Any, shape_recognizer: Optional[ShapeRecognizer] = None,
                 correction_threshold: float = DEFAULT_SHAPE_CORRECTION_THRESHOLD, debug: bool = False) -> None:
        if not 0.0 <= correction_threshold <= 1.0:
            raise ValueError("correction_threshold must be between 0.0 and 1.0")
        self.ocr_processor = ocr_processor
        self.shape_recognizer = shape_recognizer or ShapeRecognizer(debug=False)
        self.correction_threshold = correction_threshold
        self.debug = debug
        self._logger = logging.getLogger(__name__)

    def _character_score(self, text: str, confidence: float, shape: RecognitionResult) -> float:
        """Score OCR as a competing candidate, with modest glyph-shape evidence."""
        compact = text.replace(" ", "")
        if not compact:
            return 0.0
        score = confidence
        if len(compact) == 1:
            score += 0.04
            # A character-like oval need not lose merely because its contour is
            # closed; this is intentionally only a small tie-breaker.
            if compact in {"o", "O", "0"} and 0.55 <= shape.features.aspect_ratio <= 1.45:
                score += 0.06
        return min(1.0, score)

    def analyze(self, image: np.ndarray, shape_correction_enabled: bool = True) -> AutomaticRecognitionResult:
        """Evaluate the same immutable drawing image with both existing systems."""
        shape = self.shape_recognizer.recognize(image)
        # OCR's established public contract is path based.  Use a short-lived
        # PNG rather than changing that contract or duplicating preprocessing.
        with NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            image_path = Path(temporary.name)
        try:
            import cv2
            cv2.imwrite(str(image_path), image)
            ocr = self.ocr_processor.extract_text_with_confidence(image_path)
        finally:
            image_path.unlink(missing_ok=True)
        text = str(ocr.get("text", "")).strip()
        ocr_confidence = float(ocr.get("confidence", 0.0))
        valid_text = bool(text and text not in {"No text recognized"} and not text.startswith(("[OCR Error]", "Empty drawing")))
        character_score = self._character_score(text, ocr_confidence, shape) if valid_text else 0.0
        shape_score = shape.confidence

        # Strong, closed geometry wins unless OCR has materially stronger
        # evidence. This prevents circles and triangles becoming O/A while
        # allowing actual characters drawn with multiple strokes to remain text.
        shape_valid = shape.shape != ShapeType.UNKNOWN and shape_score >= self.correction_threshold
        # Shape Correct changes arbitration policy, not the availability of OCR:
        # while it is off a credible character candidate wins a round-contour
        # tie; while it is on a strong geometric candidate is allowed a small
        # margin to enable intentional diagram correction.
        margin = (RECOGNITION_SETTINGS.shape_correct_text_margin if shape_correction_enabled
                  else RECOGNITION_SETTINGS.shape_text_margin)
        is_round = shape.shape in {ShapeType.CIRCLE, ShapeType.ELLIPSE}
        prefer_shape = shape_valid and (not valid_text or shape_score >= character_score + margin)
        if is_round and not shape_correction_enabled and valid_text and len(text.replace(" ", "")) == 1:
            prefer_shape = False
        if shape.shape == ShapeType.CIRCLE and shape_score < RECOGNITION_SETTINGS.circle_candidate_threshold:
            prefer_shape = False
        if prefer_shape:
            result = AutomaticRecognitionResult("shape", shape.shape.value, shape.confidence,
                                                "ShapeRecognizer", shape)
        elif valid_text:
            kind: Literal["character", "text"] = "character" if len(text.replace(" ", "")) == 1 else "text"
            result = AutomaticRecognitionResult(kind, text, ocr_confidence, "Tesseract OCR", text=text)
        elif shape.shape != ShapeType.UNKNOWN:
            result = AutomaticRecognitionResult("unknown", shape.shape.value, shape.confidence,
                                                "ShapeRecognizer", shape)
        else:
            result = AutomaticRecognitionResult("unknown", "unknown", 0.0, "none", shape)
        if self.debug:
            self._logger.debug(
                "Recognition candidates: OCR=%r score=%.2f raw_confidence=%.2f; "
                "shape=%s score=%.2f; Shape Correct=%s; final=%s correction_eligible=%s",
                text, character_score, ocr_confidence, shape.shape.value, shape_score,
                shape_correction_enabled, result.detected, prefer_shape,
            )
        return result

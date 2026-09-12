"""Central, environment-aware configuration for recognition behaviour."""

from dataclasses import dataclass
import os


def _fraction(name: str, default: float) -> float:
    """Read a normalized confidence setting without accepting invalid values."""
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if 0.0 <= value <= 1.0 else default


def _milliseconds(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if 100 <= value <= 5000 else default


def _degrees(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if 5.0 <= value <= 60.0 else default


@dataclass(frozen=True)
class RecognitionSettings:
    """One source of truth for local recognition confidence and grouping."""

    shape_recognition_threshold: float = _fraction("SHAPE_RECOGNITION_THRESHOLD", 0.70)
    shape_correction_threshold: float = _fraction("SHAPE_CORRECTION_THRESHOLD", 0.70)
    text_recognition_threshold: float = _fraction("TEXT_RECOGNITION_THRESHOLD", 0.70)
    # Candidate arbitration is kept here so the UI, automatic recognizer, and
    # tests have one consistent definition of a "strong" geometric drawing.
    circle_candidate_threshold: float = _fraction("CIRCLE_CANDIDATE_THRESHOLD", 0.78)
    shape_text_margin: float = _fraction("SHAPE_TEXT_MARGIN", 0.12)
    shape_correct_text_margin: float = _fraction("SHAPE_CORRECT_TEXT_MARGIN", 0.04)
    polygon_angle_tolerance: float = _degrees("POLYGON_ANGLE_TOLERANCE", 22.0)
    stroke_group_delay_ms: int = _milliseconds("STROKE_GROUP_DELAY_MS", 650)


RECOGNITION_SETTINGS = RecognitionSettings()

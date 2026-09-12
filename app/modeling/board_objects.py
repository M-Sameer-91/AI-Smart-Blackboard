"""Structured board records shared by rendering, semantic AI, and future tools."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


BoundingBox = Tuple[int, int, int, int]
BoardObjectKind = Literal["shape", "character", "text", "math", "graph", "model3d", "unknown"]


@dataclass(frozen=True)
class BoardObject:
    """An immutable recognition record; it never owns or mutates canvas pixels."""

    kind: BoardObjectKind
    label: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    geometry: Optional[Any] = None
    source_stroke_range: Optional[Tuple[int, int]] = None
    corrected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        suffix = " (regularized)" if self.corrected else ""
        return f"{self.kind}: {self.label} ({self.confidence:.0%}){suffix}"

"""Generate clean, canvas-ready geometry from a recognized 2D shape."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from app.modeling.shape_recognizer import ShapeFeatures, ShapeType


Point = Tuple[float, float]


@dataclass(frozen=True)
class CorrectedShape:
    """A normalized geometric path retained by the canvas stroke model."""
    shape: ShapeType
    points: List[Point]
    center: Point
    dimensions: Tuple[float, float]


class ShapeCorrector:
    """Convert ShapeRecognizer output into clean, extensible geometric paths."""

    _REGULAR_SIDES = {
        ShapeType.SQUARE: 4, ShapeType.PENTAGON: 5, ShapeType.HEXAGON: 6,
        ShapeType.HEPTAGON: 7, ShapeType.OCTAGON: 8, ShapeType.NONAGON: 9,
        ShapeType.DECAGON: 10, ShapeType.UNDECAGON: 11, ShapeType.DODECAGON: 12,
    }

    def correct(self, shape: ShapeType, features: ShapeFeatures,
                contour_points: Sequence[Point] | None = None,
                vertices: Sequence[Point] | None = None) -> CorrectedShape:
        """Return a closed/open ideal path in the canvas coordinate system."""
        x, y, width, height = features.bounding_rect
        center = (x + width / 2, y + height / 2)
        contour = list(contour_points or [])
        polygon_vertices = list(vertices or [])

        if shape == ShapeType.CIRCLE:
            return self._circle(shape, contour, center, width, height)
        if shape == ShapeType.ELLIPSE:
            return self._ellipse(contour, center, width, height)
        if shape == ShapeType.LINE:
            return self._line(contour, center, width, height)
        if shape == ShapeType.STAR:
            return self._star(polygon_vertices or contour, center, width, height)
        if shape in self._REGULAR_SIDES:
            return self._regular_polygon(shape, self._REGULAR_SIDES[shape], polygon_vertices or contour, center, width, height)
        if shape == ShapeType.RECTANGLE and len(contour) >= 4:
            box = cv2.boxPoints(cv2.minAreaRect(np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))))
            points = [(float(point[0]), float(point[1])) for point in box]
            return CorrectedShape(shape, points + [points[0]], center, (width, height))
        if shape in {ShapeType.TRIANGLE, ShapeType.TRAPEZOID, ShapeType.PARALLELOGRAM,
                     ShapeType.RHOMBUS, ShapeType.KITE} and len(polygon_vertices) >= 3:
            return CorrectedShape(shape, polygon_vertices + [polygon_vertices[0]], center, (width, height))
        # Extensible safe fallback for any future polygon-like ShapeType.
        return self._regular_polygon(shape, max(3, len(polygon_vertices) or 4), polygon_vertices or contour, center, width, height)

    @staticmethod
    def _closed_ellipse(center: Point, radius_x: float, radius_y: float, angle: float = 0.0) -> List[Point]:
        cx, cy = center
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        points = []
        for index in range(97):
            theta = index * 2 * math.pi / 96
            px, py = radius_x * math.cos(theta), radius_y * math.sin(theta)
            points.append((cx + px * cos_a - py * sin_a, cy + px * sin_a + py * cos_a))
        return points

    def _circle(self, shape: ShapeType, contour: Sequence[Point], center: Point, width: float, height: float) -> CorrectedShape:
        if len(contour) >= 3:
            cv_contour = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
            (cx, cy), radius = cv2.minEnclosingCircle(cv_contour)
            center = (float(cx), float(cy))
        else:
            radius = min(width, height) / 2
        radius = max(1.0, float(radius))
        return CorrectedShape(shape, self._closed_ellipse(center, radius, radius), center, (2 * radius, 2 * radius))

    def _ellipse(self, contour: Sequence[Point], center: Point, width: float, height: float) -> CorrectedShape:
        angle = 0.0
        radius_x, radius_y = max(1.0, width / 2), max(1.0, height / 2)
        if len(contour) >= 5:
            cv_contour = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
            (cx, cy), (axis_a, axis_b), degrees = cv2.fitEllipse(cv_contour)
            center = (float(cx), float(cy))
            radius_x, radius_y = max(1.0, axis_a / 2), max(1.0, axis_b / 2)
            angle = math.radians(float(degrees))
        return CorrectedShape(ShapeType.ELLIPSE, self._closed_ellipse(center, radius_x, radius_y, angle), center, (2 * radius_x, 2 * radius_y))

    @staticmethod
    def _line(contour: Sequence[Point], center: Point, width: float, height: float) -> CorrectedShape:
        if len(contour) >= 2:
            return CorrectedShape(ShapeType.LINE, [tuple(contour[0]), tuple(contour[-1])], center, (width, height))
        return CorrectedShape(ShapeType.LINE, [(center[0] - width / 2, center[1]), (center[0] + width / 2, center[1])], center, (width, height))

    def _regular_polygon(self, shape: ShapeType, sides: int, contour: Sequence[Point], center: Point, width: float, height: float) -> CorrectedShape:
        radius = max(1.0, min(width, height) / 2)
        angle = -math.pi / 2
        if contour:
            angle = math.atan2(contour[0][1] - center[1], contour[0][0] - center[0])
        points = [(center[0] + radius * math.cos(angle + index * 2 * math.pi / sides),
                   center[1] + radius * math.sin(angle + index * 2 * math.pi / sides)) for index in range(sides)]
        return CorrectedShape(shape, points + [points[0]], center, (2 * radius, 2 * radius))

    def _star(self, contour: Sequence[Point], center: Point, width: float, height: float) -> CorrectedShape:
        outer = max(1.0, min(width, height) / 2)
        angle = math.atan2(contour[0][1] - center[1], contour[0][0] - center[0]) if contour else -math.pi / 2
        points = []
        for index in range(10):
            radius = outer if index % 2 == 0 else outer * 0.46
            theta = angle + index * math.pi / 5
            points.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
        return CorrectedShape(ShapeType.STAR, points + [points[0]], center, (2 * outer, 2 * outer))

"""Safe, deterministic parsing for recognized mathematical expressions."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MathRecognition:
    expression: str
    normalized: str
    confidence: float
    parsed: bool
    error: Optional[str] = None


class MathEngine:
    """Use SymPy when installed; never execute handwritten text as Python."""

    def analyze(self, text: str, confidence: float) -> MathRecognition:
        cleaned = text.strip()
        normalized = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")
        if not normalized:
            return MathRecognition(cleaned, normalized, 0.0, False, "empty expression")
        try:
            from sympy import Eq
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application,
                parse_expr,
                standard_transformations,
            )
            transformations = standard_transformations + (implicit_multiplication_application,)
            if "=" in normalized:
                left, right = normalized.split("=", 1)
                Eq(parse_expr(left, transformations=transformations),
                   parse_expr(right, transformations=transformations))
            else:
                parse_expr(normalized, transformations=transformations)
        except Exception as exc:
            return MathRecognition(cleaned, normalized, confidence, False, type(exc).__name__)
        return MathRecognition(cleaned, normalized, confidence, True)

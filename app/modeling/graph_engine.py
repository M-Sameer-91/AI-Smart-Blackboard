"""Deterministic graph data and Matplotlib figures from validated equations."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class GraphData:
    kind: Literal["explicit", "implicit"]
    expression: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray


class GraphEngine:
    """Plot only parsed SymPy expressions; never evaluate OCR text as Python."""

    def build(self, normalized_expression: str, lower: float = -10, upper: float = 10,
              samples: int = 401) -> GraphData:
        from sympy import Eq, Symbol, solve
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (implicit_multiplication_application,)
        x, y = Symbol("x"), Symbol("y")
        if "=" in normalized_expression:
            left, right = normalized_expression.split("=", 1)
            left_expression = parse_expr(left, transformations=transformations)
            right_expression = parse_expr(right, transformations=transformations)
            equation = Eq(left_expression, right_expression)
            # y=f(x) is naturally an explicit graph. Equations that contain y
            # on both sides (for example x**2 + y**2 = 25) must remain implicit
            # or solving for y would discard one branch of the curve.
            y_polynomial = (left_expression - right_expression).as_poly(y)
            y_degree = y_polynomial.degree() if y_polynomial is not None else None
            explicit_candidate = (
                y in left_expression.free_symbols
                and y not in right_expression.free_symbols
                and y_degree == 1
            )
        else:
            equation = Eq(y, parse_expr(normalized_expression, transformations=transformations))
            explicit_candidate = True
        axis = np.linspace(lower, upper, samples)
        solutions = solve(equation, y) if explicit_candidate else []
        if solutions:
            from sympy import lambdify
            fn = lambdify(x, solutions[0], modules=["numpy"])
            values = np.asarray(fn(axis), dtype=float)
            if values.ndim == 0:
                values = np.full_like(axis, values, dtype=float)
            return GraphData("explicit", normalized_expression, axis, values, values)

        from sympy import lambdify
        fn = lambdify((x, y), equation.lhs - equation.rhs, modules=["numpy"])
        grid_x, grid_y = np.meshgrid(axis, axis)
        values = np.asarray(fn(grid_x, grid_y), dtype=float)
        return GraphData("implicit", normalized_expression, grid_x, grid_y, values)

    @staticmethod
    def figure(graph: GraphData) -> Any:
        """Create a standard Matplotlib figure for an existing Tk/GUI caller."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(5, 4), dpi=100)
        axes = figure.add_subplot(111)
        if graph.kind == "explicit":
            axes.plot(graph.x, graph.y)
        else:
            axes.contour(graph.x, graph.y, graph.values, levels=[0])
        axes.axhline(0, color="black", linewidth=0.7)
        axes.axvline(0, color="black", linewidth=0.7)
        axes.grid(True, alpha=0.25)
        axes.set_title(graph.expression)
        return figure

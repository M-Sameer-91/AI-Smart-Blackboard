"""Deterministic graph-generation coverage for explicit and implicit equations."""

import unittest

from app.modeling.graph_engine import GraphEngine


class GraphEngineTests(unittest.TestCase):
    def test_explicit_equation_generates_line_data(self):
        graph = GraphEngine().build("y = 2*x + 1")
        self.assertEqual(graph.kind, "explicit")
        self.assertEqual(graph.x.shape, graph.y.shape)
        self.assertAlmostEqual(float(graph.y[len(graph.y) // 2]), 1.0, places=6)

    def test_implicit_circle_generates_contour_grid(self):
        graph = GraphEngine().build("x**2 + y**2 = 25")
        self.assertEqual(graph.kind, "implicit")
        self.assertEqual(graph.values.shape, graph.x.shape)
        self.assertEqual(graph.values.shape, graph.y.shape)


if __name__ == "__main__":
    unittest.main()

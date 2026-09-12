"""Canvas-model tests for visible, undoable auto-correction."""

import unittest

from app.modeling.shape_recognizer import ShapeFeatures, ShapeType
from app.ui.canvas import DrawMode, DrawingCanvas, StrokeData


class _VisibleCanvas:
    def __init__(self):
        self.deleted = 0
        self.lines = []

    def delete(self, _tag):
        self.deleted += 1
        self.lines.clear()

    def create_line(self, *args, **kwargs):
        self.lines.append((args, kwargs))


def _board(strokes):
    board = object.__new__(DrawingCanvas)
    board._canvas = _VisibleCanvas()
    board._stroke_history = strokes
    board._undo_states = []
    board._redo_states = []
    return board


class CanvasAutoCorrectionTests(unittest.TestCase):
    def test_circle_replaces_only_target_group_and_requests_visible_redraw(self):
        rough = StrokeData([(100, 100), (170, 95), (200, 150), (175, 200), (100, 205), (95, 150)], "white", 3)
        text = StrokeData([(300, 300), (330, 305)], "white", 3)
        board = _board([rough, text])
        features = ShapeFeatures(bounding_rect=(95, 95, 110, 110))

        self.assertTrue(board.replace_strokes_with_shape(0, 1, ShapeType.CIRCLE, features))
        self.assertEqual(len(board._stroke_history), 2)
        self.assertTrue(board._stroke_history[0].is_corrected)
        self.assertEqual(board._stroke_history[1].points, text.points)
        self.assertEqual(board._stroke_history[0].points[0], board._stroke_history[0].points[-1])
        self.assertEqual(board._canvas.deleted, 1)
        self.assertGreater(len(board._canvas.lines), 90)
        self.assertTrue(all(call[1]["smooth"] is False for call in board._canvas.lines[:96]))

    def test_undo_and_redo_restore_rough_then_corrected_shape(self):
        rough = StrokeData([(10, 10), (90, 12), (88, 90), (12, 88), (10, 10)], "white", 3)
        board = _board([rough])
        features = ShapeFeatures(bounding_rect=(10, 10, 80, 80))
        board.replace_strokes_with_shape(0, 1, ShapeType.SQUARE, features)
        corrected = list(board._stroke_history[0].points)

        self.assertTrue(board.undo_last_stroke())
        self.assertFalse(board._stroke_history[0].is_corrected)
        self.assertEqual(board._stroke_history[0].points, rough.points)
        self.assertTrue(board.redo_last_stroke())
        self.assertTrue(board._stroke_history[0].is_corrected)
        self.assertEqual(board._stroke_history[0].points, corrected)

    def test_line_uses_original_endpoints(self):
        rough_line = StrokeData([(20, 45), (35, 44), (80, 49), (140, 55)], "white", 3, DrawMode.PEN)
        board = _board([rough_line])
        self.assertTrue(board.replace_strokes_with_shape(0, 1, ShapeType.LINE, ShapeFeatures(bounding_rect=(20, 44, 120, 12))))
        self.assertEqual(board._stroke_history[0].points, [(20, 45), (140, 55)])

    def test_regular_polygons_and_star_are_inserted_as_clean_closed_vectors(self):
        for shape, expected_edges in ((ShapeType.PENTAGON, 5), (ShapeType.HEXAGON, 6), (ShapeType.OCTAGON, 8), (ShapeType.STAR, 10)):
            with self.subTest(shape=shape.value):
                board = _board([StrokeData([(100, 50), (150, 100), (100, 150)], "white", 3)])
                self.assertTrue(board.replace_strokes_with_shape(
                    0, 1, shape, ShapeFeatures(bounding_rect=(50, 50, 100, 100)), [(100, 50)]
                ))
                points = board._stroke_history[0].points
                self.assertEqual(len(points), expected_edges + 1)
                self.assertEqual(points[0], points[-1])
                self.assertTrue(board._stroke_history[0].is_corrected)


if __name__ == "__main__":
    unittest.main()

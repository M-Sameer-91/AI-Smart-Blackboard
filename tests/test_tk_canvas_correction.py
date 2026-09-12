"""Integration test proving corrected strokes render on the real Tk Canvas."""

import tkinter as tk
import unittest

from app.modeling.shape_recognizer import ShapeFeatures, ShapeType
from app.ui.canvas import DrawingCanvas


class TkCanvasCorrectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        self.root.withdraw()
        self.board = DrawingCanvas(self.root, width=400, height=300, pen_size=3)
        self.board.get_canvas().pack()
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_rough_circle_is_replaced_by_clean_lines_on_visible_widget(self):
        for x, y in ((100, 80), (150, 65), (195, 110), (180, 165), (125, 180), (90, 130), (100, 80)):
            if (x, y) == (100, 80):
                self.board.start_draw(x, y)
            else:
                self.board.draw(x, y)
        self.board.stop_draw()
        self.assertTrue(self.board.replace_strokes_with_shape(
            0, 1, ShapeType.CIRCLE, ShapeFeatures(bounding_rect=(90, 65, 105, 115))
        ))
        self.root.update_idletasks()

        items = self.board.get_canvas().find_all()
        self.assertGreaterEqual(len(items), 96)
        self.assertTrue(all(self.board.get_canvas().type(item) == "line" for item in items))
        self.assertEqual(self.board.get_canvas().itemcget(items[0], "smooth"), "0")
        self.assertTrue(self.board.get_stroke_history()[0].is_corrected)


if __name__ == "__main__":
    unittest.main()

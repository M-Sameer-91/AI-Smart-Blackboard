"""Embedded, interactive trimesh viewer for the Tk blackboard UI."""

from typing import Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MeshViewer:
    """Renders a trimesh mesh with Matplotlib's built-in mouse rotation."""

    def __init__(self, parent: tk.Widget) -> None:
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor="#1a1a1a")
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self._mesh = None
        self._configure_axes("3D preview will appear here")

    def _configure_axes(self, title: str) -> None:
        self.axes.clear()
        self.axes.set_facecolor("#1a1a1a")
        self.axes.set_title(title, color="white", fontsize=10)
        self.axes.tick_params(colors="#bbbbbb", labelsize=7)
        self.axes.set_xlabel("X", color="#bbbbbb")
        self.axes.set_ylabel("Y", color="#bbbbbb")
        self.axes.set_zlabel("Z", color="#bbbbbb")

    def show_mesh(self, mesh, title: str = "Generated 3D model") -> None:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Cannot render an empty mesh")
        self._mesh = mesh.copy()
        self._configure_axes(title + " — drag to rotate")
        triangles = self._mesh.vertices[self._mesh.faces]
        collection = Poly3DCollection(
            triangles, facecolors="#4ea3ff", edgecolors="#d9ecff", linewidths=0.25, alpha=0.92
        )
        self.axes.add_collection3d(collection)
        self._fit_mesh()
        self.axes.view_init(elev=25, azim=-55)
        self.canvas.draw_idle()

    def _fit_mesh(self) -> None:
        bounds = self._mesh.bounds
        center = bounds.mean(axis=0)
        span = max(float(np.max(bounds[1] - bounds[0])), 1.0)
        half = span * 0.65
        self.axes.set_xlim(center[0] - half, center[0] + half)
        self.axes.set_ylim(center[1] - half, center[1] + half)
        self.axes.set_zlim(center[2] - half, center[2] + half)
        self.axes.set_box_aspect((1, 1, 1))

    def reset_view(self) -> None:
        if self._mesh is not None:
            self._fit_mesh()
            self.axes.view_init(elev=25, azim=-55)
            self.canvas.draw_idle()

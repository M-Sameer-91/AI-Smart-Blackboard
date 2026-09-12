import tkinter as tk
from tkinter import Canvas
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from app.modeling.shape_corrector import ShapeCorrector


class DrawMode(Enum):
    """Drawing modes supported by the canvas."""
    PEN = "pen"
    ERASER = "eraser"


@dataclass
class StrokeData:
    """
    Data structure for storing stroke information.
    Designed for future AI analysis, OCR, and shape recognition.
    """
    points: List[Tuple[float, float]] = field(default_factory=list)
    color: str = "white"
    width: float = 2.0
    mode: DrawMode = DrawMode.PEN
    timestamp: float = 0.0
    is_corrected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stroke data to dictionary for serialization."""
        return {
            "points": self.points,
            "color": self.color,
            "width": self.width,
            "mode": self.mode.value,
            "timestamp": self.timestamp
        }


class DrawingCanvas:
    """
    Professional drawing canvas for AI Smart Blackboard.
    
    Manages all drawing operations with support for:
    - Freehand drawing with smooth strokes
    - Configurable pen color and size
    - Eraser mode infrastructure
    - Stroke data storage for future AI integration
    - Complete canvas clearing
    - Undo/Redo infrastructure
    
    This class is designed to be UI-agnostic and can be integrated
    with any parent window or frame.
    """
    
    def __init__(
        self, 
        parent: tk.Widget,
        width: int = 800,
        height: int = 600,
        bg: str = "black",
        pen_color: str = "white",
        pen_size: float = 2.0
    ) -> None:
        """
        Initialize the drawing canvas with specified properties.
        
        Args:
            parent: Parent widget (Tk, Toplevel, or Frame)
            width: Canvas width in pixels
            height: Canvas height in pixels
            bg: Background color (default: black)
            pen_color: Default pen color (default: white)
            pen_size: Default pen size in pixels (default: 2.0)
        """
        self._parent = parent
        self._width = width
        self._height = height
        self._bg = bg
        
        # Drawing state
        self._is_drawing: bool = False
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._current_stroke: Optional[StrokeData] = None
        
        # Pen properties
        self._pen_color: str = pen_color
        self._pen_size: float = pen_size
        self._mode: DrawMode = DrawMode.PEN
        
        # Stroke history for future AI analysis
        self._stroke_history: List[StrokeData] = []
        self._undo_states: List[List[StrokeData]] = []
        self._redo_states: List[List[StrokeData]] = []
        self._stroke_finished_callback: Optional[Callable[[int], None]] = None
        
        # Create the tkinter canvas
        self._canvas: Canvas = Canvas(
            self._parent,
            width=self._width,
            height=self._height,
            bg=self._bg,
            highlightthickness=0,
            cursor="pencil"
        )
        
        # Bind mouse events
        self._bind_events()
        
        # Configure canvas for smooth drawing
        self._canvas.configure(relief="flat")
        
    def _bind_events(self) -> None:
        """Bind mouse events to the canvas."""
        self._canvas.bind("<ButtonPress-1>", self._on_button_press)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_button_release)
        
    def _on_button_press(self, event: tk.Event) -> None:
        """
        Handle mouse button press event.
        Starts a new drawing stroke.
        """
        self._is_drawing = True
        self._last_x = event.x
        self._last_y = event.y
        
        # Initialize new stroke data
        import time
        self._current_stroke = StrokeData(
            points=[(event.x, event.y)],
            color=self._pen_color if self._mode == DrawMode.PEN else self._bg,
            width=self._pen_size,
            mode=self._mode,
            timestamp=time.time()
        )
        
        # Draw initial point for immediate feedback
        self._canvas.create_oval(
            event.x - self._pen_size/2,
            event.y - self._pen_size/2,
            event.x + self._pen_size/2,
            event.y + self._pen_size/2,
            fill=self._pen_color if self._mode == DrawMode.PEN else self._bg,
            outline=""
        )
        
    def _on_mouse_drag(self, event: tk.Event) -> None:
        """
        Handle mouse drag event.
        Draws smooth strokes using line segments.
        """
        if not self._is_drawing or self._last_x is None or self._last_y is None:
            return
            
        x, y = event.x, event.y
        
        # Determine color based on mode
        color = self._pen_color if self._mode == DrawMode.PEN else self._bg
        
        # Draw line segment for smooth stroke
        self._canvas.create_line(
            self._last_x, self._last_y,
            x, y,
            fill=color,
            width=self._pen_size,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=4
        )
        
        # Store point in current stroke data
        if self._current_stroke is not None:
            self._current_stroke.points.append((x, y))
        
        # Update last position
        self._last_x = x
        self._last_y = y
        
    def _on_button_release(self, event: tk.Event) -> None:
        """
        Handle mouse button release event.
        Finalizes the current stroke and stores it for AI analysis.
        """
        if self._is_drawing and self._current_stroke is not None:
            # Finalize stroke
            if self._current_stroke.points:
                self._save_undo_state()
                self._stroke_history.append(self._current_stroke)
                if self._stroke_finished_callback:
                    self._stroke_finished_callback(len(self._stroke_history))
            
        # Reset drawing state
        self._is_drawing = False
        self._last_x = None
        self._last_y = None
        self._current_stroke = None
        
    def start_draw(self, x: float, y: float) -> None:
        """
        Public method to start drawing at specified coordinates.
        Useful for programmatic drawing or integration.
        
        Args:
            x: X-coordinate to start drawing
            y: Y-coordinate to start drawing
        """
        # Create a synthetic event
        event = tk.Event()
        event.x = x
        event.y = y
        self._on_button_press(event)
        
    def draw(self, x: float, y: float) -> None:
        """
        Public method to draw to specified coordinates.
        Useful for programmatic drawing or integration.
        
        Args:
            x: X-coordinate to draw to
            y: Y-coordinate to draw to
        """
        if not self._is_drawing:
            return
        event = tk.Event()
        event.x = x
        event.y = y
        self._on_mouse_drag(event)
        
    def stop_draw(self) -> None:
        """
        Public method to stop the current drawing operation.
        Useful for programmatic drawing or integration.
        """
        self._is_drawing = False
        self._last_x = None
        self._last_y = None
        if self._current_stroke is not None and self._current_stroke.points:
            self._save_undo_state()
            self._stroke_history.append(self._current_stroke)
            if self._stroke_finished_callback:
                self._stroke_finished_callback(len(self._stroke_history))
        self._current_stroke = None
        
    def clear_canvas(self) -> None:
        """
        Clear the entire drawing canvas.
        Removes all drawn content while preserving canvas configuration.
        """
        self._canvas.delete("all")
        self._stroke_history.clear()
        self._undo_states.clear()
        self._redo_states.clear()
        self._current_stroke = None
        self._is_drawing = False
        self._last_x = None
        self._last_y = None
        
    def set_pen_color(self, color: str) -> None:
        """
        Set the pen color for drawing.
        
        Args:
            color: Color name or hex code (e.g., "white", "#FF0000")
        """
        self._pen_color = color
        
    def set_pen_size(self, size: float) -> None:
        """
        Set the pen size for drawing.
        
        Args:
            size: Pen size in pixels (must be > 0)
        """
        if size <= 0:
            raise ValueError("Pen size must be greater than 0")
        self._pen_size = size
        
    def set_mode(self, mode: DrawMode) -> None:
        """
        Set the drawing mode.
        
        Args:
            mode: DrawMode.PEN or DrawMode.ERASER
        """
        self._mode = mode
        # Update cursor based on mode
        if mode == DrawMode.PEN:
            self._canvas.configure(cursor="pencil")
        elif mode == DrawMode.ERASER:
            self._canvas.configure(cursor="circle")
            
    def get_canvas(self) -> Canvas:
        """
        Get the underlying tkinter Canvas widget.
        
        Returns:
            tkinter.Canvas: The canvas widget for packing/placement
        """
        return self._canvas
    
    def get_stroke_history(self) -> List[StrokeData]:
        """
        Get the complete stroke history for AI analysis.
        
        Returns:
            List[StrokeData]: List of all strokes drawn on the canvas
        """
        return self._stroke_history.copy()

    def set_stroke_finished_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Register a notification invoked after each completed pen stroke."""
        self._stroke_finished_callback = callback

    def _save_undo_state(self) -> None:
        """Store an immutable-enough copy before a user-visible board change."""
        self._undo_states.append([
            StrokeData(list(stroke.points), stroke.color, stroke.width, stroke.mode, stroke.timestamp, stroke.is_corrected)
            for stroke in self._stroke_history
        ])
        self._redo_states.clear()
        # Bound memory while retaining normal interactive undo behaviour.
        if len(self._undo_states) > 100:
            self._undo_states.pop(0)

    def replace_strokes_with_shape(self, start_index: int, end_index: int, shape: Any, features: Any,
                                   raw_vertices: Optional[List[Tuple[float, float]]] = None,
                                   contour_points: Optional[List[Tuple[float, float]]] = None) -> bool:
        """Replace a completed stroke group with a clean vector-like stroke.

        The source strokes remain recoverable through ``undo_last_stroke``.
        ``features`` is deliberately attribute-based because ShapeFeatures is a
        dataclass, not a dictionary.
        """
        if start_index < 0 or end_index <= start_index or end_index > len(self._stroke_history):
            return False
        self._save_undo_state()
        shape_name = getattr(shape, "value", str(shape)).lower()
        x, y, width, height = getattr(features, "bounding_rect", (0, 0, 0, 0))
        if width <= 1 or height <= 1:
            return False
        source_points = [point for stroke in self._stroke_history[start_index:end_index] for point in stroke.points]
        corrected = ShapeCorrector().correct(
            shape, features, contour_points or (source_points if shape_name == "line" else None), raw_vertices
        )
        points = corrected.points
        source = self._stroke_history[start_index]
        # Preserve entries before and after this recognition group: an
        # asynchronous result must never erase another object drawn later.
        self._stroke_history[start_index:end_index] = [
            StrokeData(points, source.color, source.width, DrawMode.PEN, source.timestamp, is_corrected=True)
        ]
        self._canvas.delete("all")
        self._redraw_strokes()
        return True
    
    def get_stroke_count(self) -> int:
        """
        Get the number of strokes stored.
        
        Returns:
            int: Total stroke count
        """
        return len(self._stroke_history)
    
    def get_stroke_data_serializable(self) -> List[Dict[str, Any]]:
        """
        Get stroke data in serializable format for AI processing.
        
        Returns:
            List[Dict]: List of stroke dictionaries ready for JSON serialization
        """
        return [stroke.to_dict() for stroke in self._stroke_history]
    
    def undo_last_stroke(self) -> bool:
        """
        Undo the last stroke drawn.
        Useful for future implementation of undo/redo functionality.
        
        Returns:
            bool: True if a stroke was undone, False if no strokes to undo
        """
        if self._undo_states:
            self._redo_states.append(self._clone_strokes(self._stroke_history))
            self._stroke_history = self._undo_states.pop()
            self._canvas.delete("all")
            self._redraw_strokes()
            return True
        if not self._stroke_history:
            return False
            
        # Remove last stroke from history
        last_stroke = self._stroke_history.pop()
        
        # Redraw all remaining strokes
        self._canvas.delete("all")
        self._redraw_strokes()
        
        return True

    def redo_last_stroke(self) -> bool:
        """Restore the most recently undone drawing or auto-correction."""
        if not self._redo_states:
            return False
        self._undo_states.append(self._clone_strokes(self._stroke_history))
        self._stroke_history = self._redo_states.pop()
        self._canvas.delete("all")
        self._redraw_strokes()
        return True

    @staticmethod
    def _clone_strokes(strokes: List[StrokeData]) -> List[StrokeData]:
        return [
            StrokeData(list(stroke.points), stroke.color, stroke.width, stroke.mode, stroke.timestamp, stroke.is_corrected)
            for stroke in strokes
        ]
    
    def _redraw_strokes(self) -> None:
        """
        Redraw all strokes from history.
        Used for undo/redo functionality and canvas refresh.
        """
        for stroke in self._stroke_history:
            if len(stroke.points) < 2:
                continue
                
            color = stroke.color
            width = stroke.width
            
            # Draw the stroke as a series of line segments
            for i in range(len(stroke.points) - 1):
                x1, y1 = stroke.points[i]
                x2, y2 = stroke.points[i + 1]
                
                self._canvas.create_line(
                    x1, y1, x2, y2,
                    fill=color,
                    width=width,
                    capstyle=tk.ROUND,
                    smooth=not stroke.is_corrected,
                    splinesteps=4 if not stroke.is_corrected else 1,
                )
    
    def resize(self, width: int, height: int) -> None:
        """
        Resize the canvas.
        
        Args:
            width: New width in pixels
            height: New height in pixels
        """
        self._width = width
        self._height = height
        self._canvas.configure(width=width, height=height)
        
    def get_canvas_size(self) -> Tuple[int, int]:
        """
        Get the current canvas size.
        
        Returns:
            Tuple[int, int]: (width, height) in pixels
        """
        # Grid expansion can make the actual board larger than the construction
        # size.  Captures must use the same coordinate system as mouse strokes.
        width = self._canvas.winfo_width()
        height = self._canvas.winfo_height()
        if width > 1 and height > 1:
            return (width, height)
        return (self._width, self._height)

"""
window.py - Main Application Window for AI Smart Blackboard

This module provides the main window class that manages the application's
UI components, layout, and communication between the toolbar and drawing canvas.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional
import customtkinter as ctk

from app.ui.canvas import DrawingCanvas, DrawMode
from app.recognition.recognizer import Recognizer


class SmartBlackboard:
    """
    Main application window for the AI Smart Blackboard.
    
    Manages the complete user interface including:
    - Window configuration and theming
    - Menu bar with File, Edit, View, and Help menus
    - Toolbar with drawing tools and actions
    - Status bar for user feedback
    - Canvas integration and communication
    - Recognition module integration
    
    This class is responsible for application management only.
    All drawing operations are delegated to DrawingCanvas.
    """
    
    def __init__(self) -> None:
        """Initialize the main application window."""
        # Configure CustomTkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("AI Smart Blackboard")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 700)
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(1, weight=1)  # Canvas row expands
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize components
        self.canvas: Optional[DrawingCanvas] = None
        self.recognizer: Optional[Recognizer] = None
        self.status_var = tk.StringVar(value="Ready")
        
        # Build UI
        self._create_menu_bar()
        self._create_toolbar()
        self._create_canvas()
        self._create_status_bar()
        
        # Initialize recognizer after canvas is created
        self._initialize_recognizer()
        
        # Bind keyboard shortcuts
        self._bind_shortcuts()
        
    def _initialize_recognizer(self) -> None:
        """
        Initialize the Recognizer with the drawing canvas.
        Handles any initialization errors gracefully.
        """
        try:
            if self.canvas:
                self.recognizer = Recognizer(self.canvas)
                self.update_status("Recognizer initialized")
            else:
                self.update_status("Error: Canvas not available for Recognizer")
        except Exception as e:
            self.update_status(f"Error initializing Recognizer: {str(e)}")
            print(f"Recognizer initialization error: {str(e)}", file=sys.stderr)
        
    def _create_menu_bar(self) -> None:
        """
        Create the application menu bar with all menus and items.
        Provides placeholders for future feature integration.
        """
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File Menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self._on_new, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self._on_open, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self._on_save, accelerator="Ctrl+S")
        file_menu.add_command(label="Export PDF", command=self._on_export_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit, accelerator="Ctrl+Q")
        
        # Edit Menu
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._on_undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._on_redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear", command=self._on_clear, accelerator="Ctrl+C")
        
        # View Menu
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self._on_zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self._on_zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self._on_zoom_reset)
        
        # Help Menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._on_about)
        
    def _create_toolbar(self) -> None:
        """
        Create the toolbar with all drawing tools and actions.
        Uses CTkButton for consistent dark theme styling.
        """
        self.toolbar = ctk.CTkFrame(self.root, height=50, corner_radius=0)
        self.toolbar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.toolbar.grid_propagate(False)
        
        # Configure toolbar grid columns
        for i in range(10):
            self.toolbar.grid_columnconfigure(i, weight=0)
        self.toolbar.grid_columnconfigure(10, weight=1)  # Spacer
        
        # Tool buttons
        self.btn_pen = ctk.CTkButton(
            self.toolbar,
            text="✏️ Pen",
            width=80,
            height=32,
            command=self._on_pen
        )
        self.btn_pen.grid(row=0, column=0, padx=5, pady=8)
        
        self.btn_eraser = ctk.CTkButton(
            self.toolbar,
            text="🧹 Eraser",
            width=80,
            height=32,
            command=self._on_eraser
        )
        self.btn_eraser.grid(row=0, column=1, padx=5, pady=8)
        
        self.btn_clear = ctk.CTkButton(
            self.toolbar,
            text="🗑️ Clear",
            width=80,
            height=32,
            command=self._on_clear
        )
        self.btn_clear.grid(row=0, column=2, padx=5, pady=8)
        
        self.btn_undo = ctk.CTkButton(
            self.toolbar,
            text="↩️ Undo",
            width=80,
            height=32,
            command=self._on_undo
        )
        self.btn_undo.grid(row=0, column=3, padx=5, pady=8)
        
        self.btn_redo = ctk.CTkButton(
            self.toolbar,
            text="↪️ Redo",
            width=80,
            height=32,
            command=self._on_redo
        )
        self.btn_redo.grid(row=0, column=4, padx=5, pady=8)
        
        # Separator
        separator = ctk.CTkFrame(self.toolbar, width=2, height=30, fg_color="gray")
        separator.grid(row=0, column=5, padx=10, pady=8)
        
        self.btn_save = ctk.CTkButton(
            self.toolbar,
            text="💾 Save",
            width=80,
            height=32,
            command=self._on_save
        )
        self.btn_save.grid(row=0, column=6, padx=5, pady=8)
        
        self.btn_recognize = ctk.CTkButton(
            self.toolbar,
            text="🔍 Recognize",
            width=100,
            height=32,
            command=self._on_recognize
        )
        self.btn_recognize.grid(row=0, column=7, padx=5, pady=8)
        
        self.btn_ai = ctk.CTkButton(
            self.toolbar,
            text="🤖 AI Explain",
            width=100,
            height=32,
            command=self._on_ai
        )
        self.btn_ai.grid(row=0, column=8, padx=5, pady=8)
        
    def _create_canvas(self) -> None:
        """
        Create and initialize the drawing canvas.
        The canvas is placed in the main window with grid expansion.
        """
        # Create a frame to hold the canvas
        canvas_frame = ctk.CTkFrame(self.root, corner_radius=0)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize DrawingCanvas
        self.canvas = DrawingCanvas(
            parent=canvas_frame,
            width=1400,
            height=700,
            bg="black",
            pen_color="white",
            pen_size=3.0
        )
        
        # Get the underlying canvas widget and pack it
        canvas_widget = self.canvas.get_canvas()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        
    def _create_status_bar(self) -> None:
        """
        Create the status bar at the bottom of the window.
        Displays current application state and user feedback.
        """
        self.status_bar = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.status_bar.grid_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=15, pady=5)
        
        # Status indicators
        self.stroke_count_label = ctk.CTkLabel(
            self.status_bar,
            text="Strokes: 0",
            font=("Segoe UI", 10),
            anchor="e"
        )
        self.stroke_count_label.pack(side="right", padx=15, pady=5)
        
    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts to application actions."""
        self.root.bind("<Control-n>", lambda e: self._on_new())
        self.root.bind("<Control-o>", lambda e: self._on_open())
        self.root.bind("<Control-s>", lambda e: self._on_save())
        self.root.bind("<Control-q>", lambda e: self._on_exit())
        self.root.bind("<Control-z>", lambda e: self._on_undo())
        self.root.bind("<Control-y>", lambda e: self._on_redo())
        self.root.bind("<Control-c>", lambda e: self._on_clear())
        self.root.bind("<Control-plus>", lambda e: self._on_zoom_in())
        self.root.bind("<Control-minus>", lambda e: self._on_zoom_out())
        
    def update_status(self, message: str) -> None:
        """
        Update the status bar with a new message.
        
        Args:
            message: Status message to display
        """
        self.status_var.set(message)
        self._update_stroke_count()
        
    def _update_stroke_count(self) -> None:
        """Update the stroke count displayed in the status bar."""
        if self.canvas:
            count = self.canvas.get_stroke_count()
            self.stroke_count_label.configure(text=f"Strokes: {count}")
        
    def _capture_canvas(self) -> Optional[str]:
        """
        Capture the current canvas content using the Recognizer.
        
        Returns:
            Optional[str]: Path to the saved image, or None if capture failed
        """
        try:
            # Check if recognizer is initialized
            if not self.recognizer:
                self.update_status("Error: Recognizer not initialized")
                return None
                
            # Capture the canvas
            image_path = self.recognizer.capture_canvas()
            
            if image_path:
                self.update_status(f"Image saved successfully: {image_path.name}")
                return str(image_path)
            else:
                self.update_status("Capture failed: Unable to capture canvas")
                return None
                
        except Exception as e:
            error_msg = f"Capture failed: {str(e)}"
            self.update_status(error_msg)
            print(f"Canvas capture error: {str(e)}", file=sys.stderr)
            return None
        
    # ======================== Toolbar Callbacks ========================
    
    def _on_pen(self) -> None:
        """Handle pen tool selection."""
        if self.canvas:
            self.canvas.set_mode(DrawMode.PEN)
            self.canvas.set_pen_color("white")
            self.update_status("Pen Selected")
        
    def _on_eraser(self) -> None:
        """Handle eraser tool selection."""
        if self.canvas:
            self.canvas.set_mode(DrawMode.ERASER)
            self.update_status("Eraser Selected")
        
    def _on_clear(self) -> None:
        """Handle clear canvas action."""
        if self.canvas:
            self.canvas.clear_canvas()
            self.update_status("Canvas Cleared")
        
    def _on_undo(self) -> None:
        """Handle undo action."""
        if self.canvas:
            if self.canvas.undo_last_stroke():
                self.update_status("Undo Successful")
            else:
                self.update_status("Nothing to Undo")
        
    def _on_redo(self) -> None:
        """Handle redo action (placeholder)."""
        self.update_status("Redo - Not Implemented Yet")
        
    def _on_save(self) -> None:
        """Handle save action (placeholder)."""
        self.update_status("Saving...")
        
    def _on_recognize(self) -> None:
        """Handle recognition action - captures the canvas using Recognizer."""
        self.update_status("Capturing whiteboard...")
        
        # Perform the capture
        image_path = self._capture_canvas()
        
        # If capture failed, status is already updated by _capture_canvas
        if image_path:
            # Additional notification for successful capture
            print(f"Canvas captured and saved to: {image_path}")
        
    def _on_ai(self) -> None:
        """Handle AI analysis action (placeholder)."""
        self.update_status("AI module not implemented")
        
    # ======================== Menu Callbacks ========================
    
    def _on_new(self) -> None:
        """Handle new file action (placeholder)."""
        if self.canvas:
            self.canvas.clear_canvas()
            self.update_status("New Board Created")
        
    def _on_open(self) -> None:
        """Handle open file action (placeholder)."""
        self.update_status("Open - Not Implemented Yet")
        
    def _on_export_pdf(self) -> None:
        """Handle export PDF action (placeholder)."""
        self.update_status("Export PDF - Not Implemented Yet")
        
    def _on_exit(self) -> None:
        """Handle exit action with confirmation."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()
            self.root.destroy()
        
    def _on_zoom_in(self) -> None:
        """Handle zoom in action (placeholder)."""
        self.update_status("Zoom In - Not Implemented Yet")
        
    def _on_zoom_out(self) -> None:
        """Handle zoom out action (placeholder)."""
        self.update_status("Zoom Out - Not Implemented Yet")
        
    def _on_zoom_reset(self) -> None:
        """Handle reset zoom action (placeholder)."""
        self.update_status("Zoom Reset - Not Implemented Yet")
        
    def _on_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About AI Smart Blackboard",
            "AI Smart Blackboard\n\n"
            "Version: 1.0.0\n"
            "Built with Python 3.12, CustomTkinter, and Tkinter\n\n"
            "A professional AI & Machine Learning project."
        )
        
    # ======================== Public Methods ========================
    
    def run(self) -> None:
        """Start the main application event loop."""
        self.update_status("Ready")
        self.root.mainloop()
        
    def get_root(self) -> ctk.CTk:
        """
        Get the main window root.
        
        Returns:
            ctk.CTk: The main application window
        """
        return self.root
        
    def get_canvas(self) -> Optional[DrawingCanvas]:
        """
        Get the drawing canvas instance.
        
        Returns:
            Optional[DrawingCanvas]: The canvas instance or None
        """
        return self.canvas
        
    def get_recognizer(self) -> Optional[Recognizer]:
        """
        Get the recognizer instance.
        
        Returns:
            Optional[Recognizer]: The recognizer instance or None
        """
        return self.recognizer
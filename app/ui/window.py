"""
window.py - Main Application Window for AI Smart Blackboard

This module provides the main window class that manages the application's
UI components, layout, and communication between the toolbar and drawing canvas.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import logging
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from typing import Optional
from PIL import Image, ImageDraw
import customtkinter as ctk

from app.ui.canvas import DrawingCanvas, DrawMode
from app.recognition.recognizer import Recognizer
from app.recognition.ocr import OCRProcessor
from app.recognition.classifier import ContentClassifier
from app.modeling.model_pipeline import ModelPipeline
from app.modeling.shape_recognizer import ShapeType
from app.config import RECOGNITION_SETTINGS
from app.modeling.board_objects import BoardObject
from app.modeling.graph_engine import GraphEngine
from app.recognition.math_engine import MathEngine
from app.recognition.orchestrator import RecognitionOrchestrator, UnifiedRecognitionResult
from app.ai.nvidia_client import NVIDIAProvider
from app.ai.provider import AIProvider, AIProviderError
from app.ui.mesh_viewer import MeshViewer


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
    - OCR processing and results display
    - Content classification
    - 3D model generation
    """

    # Shared by the automatic engine and the UI correction decision.
    # Confidence values are normalized to the inclusive 0.0–1.0 range.
    SHAPE_CORRECTION_THRESHOLD = RECOGNITION_SETTINGS.shape_correction_threshold
    AUTO_RECOGNITION_DELAY_MS = RECOGNITION_SETTINGS.stroke_group_delay_ms
    
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
        self.root.grid_rowconfigure(1, weight=1)  # Main content row expands
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize components
        self.canvas: Optional[DrawingCanvas] = None
        self.recognizer: Optional[Recognizer] = None
        self.ocr_processor: Optional[OCRProcessor] = None
        self.classifier: Optional[ContentClassifier] = None
        self.model_pipeline: Optional[ModelPipeline] = None
        self.mesh_viewer: Optional[MeshViewer] = None
        self.automatic_recognizer: Optional[RecognitionOrchestrator] = None
        self.ai_provider: Optional[AIProvider] = None
        self.math_engine = MathEngine()
        self.graph_engine = GraphEngine()
        self._last_automatic_result: Optional[UnifiedRecognitionResult] = None
        self.board_objects: list[BoardObject] = []
        self.shape_correction_enabled = tk.BooleanVar(value=True)
        self._recognition_after_id: Optional[str] = None
        self._pending_group_start: Optional[int] = None
        self._logger = logging.getLogger(__name__)
        self.status_var = tk.StringVar(value="Ready")
        
        # Build UI
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_content()
        self._create_status_bar()
        
        # Initialize recognizer and OCR after canvas is created
        self._initialize_recognizer()
        self._initialize_ocr()
        self._initialize_classifier()
        self._initialize_model_pipeline()
        self._initialize_automatic_recognition()
        self._initialize_ai_provider()
        
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
            
    def _initialize_ocr(self) -> None:
        """
        Initialize the OCRProcessor with Tesseract.
        """
        try:
            self.ocr_processor = OCRProcessor()
            self.update_status("OCR processor initialized (Tesseract)")
        except Exception as e:
            self.update_status(f"Error initializing OCR: {str(e)}")
            print(f"OCR initialization error: {str(e)}", file=sys.stderr)
            self.ocr_processor = None
    
    def _initialize_classifier(self) -> None:
        """
        Initialize the ContentClassifier.
        """
        try:
            self.classifier = ContentClassifier()
            self.update_status("Classifier initialized")
        except Exception as e:
            self.update_status(f"Error initializing Classifier: {str(e)}")
            print(f"Classifier initialization error: {str(e)}", file=sys.stderr)
            self.classifier = None
    
    def _initialize_model_pipeline(self) -> None:
        """
        Initialize the ModelPipeline for 3D model generation with 70% confidence threshold.
        """
        try:
            self.model_pipeline = ModelPipeline(
                confidence_threshold=RECOGNITION_SETTINGS.shape_recognition_threshold,
                output_dir="app/data/models",
                pixels_per_mm=5.0,
                default_height=20.0
            )
            self.update_status("3D Model pipeline initialized (70% confidence threshold)")
        except Exception as e:
            self.update_status(f"Error initializing 3D Model pipeline: {str(e)}")
            print(f"ModelPipeline initialization error: {str(e)}", file=sys.stderr)
            self.model_pipeline = None

    def _initialize_automatic_recognition(self) -> None:
        """Connect grouped stroke completion to the existing OCR/shape systems."""
        if not self.canvas or not self.ocr_processor:
            return
        self.automatic_recognizer = RecognitionOrchestrator(
            self.ocr_processor, classifier=self.classifier,
            correction_threshold=self.SHAPE_CORRECTION_THRESHOLD, debug=False,
        )
        self.canvas.set_stroke_finished_callback(self._on_stroke_finished)

    def _initialize_ai_provider(self) -> None:
        """Configure optional NVIDIA explanations without affecting local recognition."""
        self.ai_provider = NVIDIAProvider.from_environment()
        if self.ai_provider.is_available:
            self.update_status("NVIDIA AI explanation provider configured")
        
    def _create_menu_bar(self) -> None:
        """
        Create the application menu bar with all menus and items.
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
        """
        self.toolbar = ctk.CTkFrame(self.root, height=50, corner_radius=0)
        self.toolbar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.toolbar.grid_propagate(False)
        
        # Configure toolbar grid columns - increased for new button
        for i in range(14):
            self.toolbar.grid_columnconfigure(i, weight=0)
        self.toolbar.grid_columnconfigure(13, weight=1)  # Spacer
        
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
        separator1 = ctk.CTkFrame(self.toolbar, width=2, height=30, fg_color="gray")
        separator1.grid(row=0, column=5, padx=10, pady=8)
        
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

        self.btn_shape_correct = ctk.CTkButton(
            self.toolbar, text="Shape Correct: ON", width=135, height=32,
            command=self._toggle_shape_correction,
        )
        self.btn_shape_correct.grid(row=0, column=8, padx=5, pady=8)
        
        self.btn_ai = ctk.CTkButton(
            self.toolbar,
            text="🤖 AI Explain",
            width=100,
            height=32,
            command=self._on_ai
        )
        self.btn_ai.grid(row=0, column=9, padx=5, pady=8)
        
        # Separator
        separator2 = ctk.CTkFrame(self.toolbar, width=2, height=30, fg_color="gray")
        separator2.grid(row=0, column=10, padx=10, pady=8)
        
        # Shape → 3D Button (separate from text recognition)
        self.btn_shape_3d = ctk.CTkButton(
            self.toolbar,
            text="📐 Shape → 3D",
            width=120,
            height=32,
            command=self._on_shape_3d
        )
        self.btn_shape_3d.grid(row=0, column=11, padx=5, pady=8)

        self.btn_graph = ctk.CTkButton(
            self.toolbar, text="Graph", width=80, height=32, command=self._on_graph
        )
        self.btn_graph.grid(row=0, column=12, padx=5, pady=8)
        
    def _create_main_content(self) -> None:
        """
        Create the main content area with canvas and results panel.
        Split into 70% canvas and 30% results panel.
        """
        # Main content frame
        main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=7)  # 70% for canvas
        main_frame.grid_columnconfigure(1, weight=3)  # 30% for results panel
        
        # Create canvas (left side)
        self._create_canvas(main_frame)
        
        # Create results panel (right side)
        self._create_results_panel(main_frame)
        
    def _create_canvas(self, parent: ctk.CTkFrame) -> None:
        """
        Create and initialize the drawing canvas.
        
        Args:
            parent: The parent frame to place the canvas in
        """
        # Create a frame to hold the canvas
        canvas_frame = ctk.CTkFrame(parent, corner_radius=0)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize DrawingCanvas
        self.canvas = DrawingCanvas(
            parent=canvas_frame,
            width=980,
            height=700,
            bg="black",
            pen_color="white",
            pen_size=3.0
        )
        
        # Get the underlying canvas widget and pack it
        canvas_widget = self.canvas.get_canvas()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        
    def _create_results_panel(self, parent: ctk.CTkFrame) -> None:
        """
        Create the results panel with OCR text display and AI placeholder.
        
        Args:
            parent: The parent frame to place the panel in
        """
        # Results panel frame
        self.results_panel = ctk.CTkFrame(
            parent,
            corner_radius=0,
            fg_color=("#2b2b2b", "#1a1a1a")
        )
        self.results_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.results_panel.grid_rowconfigure(0, weight=0)  # Title
        self.results_panel.grid_rowconfigure(1, weight=2)  # OCR text
        self.results_panel.grid_rowconfigure(2, weight=0)  # Separator
        self.results_panel.grid_rowconfigure(3, weight=0)  # Category label
        self.results_panel.grid_rowconfigure(4, weight=0)  # AI placeholder
        self.results_panel.grid_rowconfigure(5, weight=2)  # interactive 3D view
        self.results_panel.grid_columnconfigure(0, weight=1)
        
        # OCR Section Title
        ocr_title = ctk.CTkLabel(
            self.results_panel,
            text="📝 Recognized Text",
            font=("Segoe UI", 16, "bold"),
            anchor="w"
        )
        ocr_title.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        
        # OCR Text Display (Scrollable)
        self.ocr_text_box = scrolledtext.ScrolledText(
            self.results_panel,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg="#2b2b2b",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            height=10
        )
        self.ocr_text_box.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        self.ocr_text_box.insert("1.0", "No text recognized yet.\nDraw something and click 'Recognize'.")
        self.ocr_text_box.config(state=tk.DISABLED)
        
        # Category Display
        self.category_label = ctk.CTkLabel(
            self.results_panel,
            text="📊 Category: Unknown",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
            text_color="#888888"
        )
        self.category_label.grid(row=3, column=0, sticky="ew", padx=15, pady=(5, 5))
        
        # Separator
        separator = ctk.CTkFrame(
            self.results_panel,
            height=2,
            fg_color="#3a3a3a"
        )
        separator.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        
        # AI Placeholder
        self.ai_placeholder = ctk.CTkLabel(
            self.results_panel,
            text="AI explanation will appear here.\nClick 'AI Explain' after recognition.",
            font=("Segoe UI", 12),
            anchor="nw",
            justify="left"
        )
        self.ai_placeholder.grid(row=4, column=0, sticky="nsew", padx=15, pady=5)

        viewer_header = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        viewer_header.grid(row=5, column=0, sticky="ew", padx=15, pady=(5, 0))
        viewer_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(viewer_header, text="3D Model Preview", font=("Segoe UI", 14, "bold"), anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        self.mesh_viewer = MeshViewer(self.results_panel)
        ctk.CTkButton(viewer_header, text="Reset view", width=80, command=self.mesh_viewer.reset_view).grid(
            row=0, column=1, padx=(5, 0)
        )
        self.mesh_viewer.widget.grid(row=6, column=0, sticky="nsew", padx=15, pady=(0, 12))
        self.results_panel.grid_rowconfigure(6, weight=3)
        
    def _create_status_bar(self) -> None:
        """
        Create the status bar at the bottom of the window.
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
            
    def _update_ocr_text(self, text: str) -> None:
        """
        Update the OCR text display with new text.
        
        Args:
            text: Text to display in the OCR box
        """
        self.ocr_text_box.config(state=tk.NORMAL)
        self.ocr_text_box.delete("1.0", tk.END)
        
        if text and text.strip() and not text.startswith("[OCR Error]"):
            self.ocr_text_box.insert("1.0", text)
        else:
            self.ocr_text_box.insert("1.0", text if text else "No text recognized.")
            
        self.ocr_text_box.config(state=tk.DISABLED)
    
    def _update_category(self, category: str, confidence: float) -> None:
        """
        Update the category display.
        
        Args:
            category: The detected category
            confidence: Confidence score
        """
        # Color mapping for categories
        colors = {
            "MATHEMATICS": "#4CAF50",  # Green
            "PHYSICS": "#2196F3",       # Blue
            "CHEMISTRY": "#FF9800",     # Orange
            "GENERAL_TEXT": "#9C27B0",  # Purple
            "DIAGRAM": "#F44336",       # Red
            "UNKNOWN": "#888888"        # Gray
        }
        
        color = colors.get(category, "#888888")
        confidence_percent = int(confidence * 100)
        
        self.category_label.configure(
            text=f"📊 Category: {category} (Confidence: {confidence_percent}%)",
            text_color=color
        )
        
    def _capture_and_recognize(self) -> None:
        """
        Execute the complete OCR pipeline:
        1. Capture canvas
        2. Run OCR
        3. Classify content
        4. Display results
        """
        try:
            # STEP 1: Update status
            self.update_status("Capturing whiteboard...")
            self.root.update_idletasks()
            
            # STEP 2: Capture canvas
            if not self.recognizer:
                self.update_status("Error: Recognizer not initialized")
                return
                
            image_path = self.recognizer.capture_canvas()
            
            # STEP 3: Check if capture succeeded
            if not image_path:
                self.update_status("Capture failed.")
                return
                
            # STEP 4: Update status
            self.update_status("Image captured. Running OCR...")
            self.root.update_idletasks()
            
            # STEP 5: Run OCR
            if not self.ocr_processor:
                self.update_status("Error: OCR processor not initialized")
                return
                
            recognized_text = self.ocr_processor.extract_text(image_path)
            
            # STEP 6: Display OCR results
            self._update_ocr_text(recognized_text)
            
            # STEP 7: Classify the text
            if self.classifier and recognized_text and not recognized_text.startswith("[OCR Error]"):
                self.update_status("Classifying content...")
                self.root.update_idletasks()
                
                classification = self.classifier.classify(recognized_text)
                category = classification.get("category", "UNKNOWN")
                confidence = classification.get("confidence", 0.0)
                
                self._update_category(category, confidence)
                
                # Update status with category
                self.update_status(f"OCR completed. Category: {category}")
            else:
                self._update_category("UNKNOWN", 0.0)
                self.update_status("OCR completed. No content to classify.")
            
        except Exception as e:
            error_msg = f"Process failed: {str(e)}"
            self.update_status(error_msg)
            self._update_ocr_text(f"Error during processing:\n{str(e)}")
            self._update_category("UNKNOWN", 0.0)
            print(f"Processing error: {str(e)}", file=sys.stderr)
    
    def _check_canvas_empty(self) -> bool:
        """
        Check if the canvas is empty (no strokes).
        
        Returns:
            bool: True if canvas is empty, False otherwise
        """
        if not self.canvas:
            return True
        
        stroke_count = self.canvas.get_stroke_count()
        return stroke_count == 0
        
    def _on_shape_3d(self) -> None:
        """
        Handle the Shape → 3D button click.
        This is for geometric shape recognition and 3D model generation.
        It is completely separate from OCR/text recognition.
        """
        # Check if pipeline is initialized
        if not self.model_pipeline:
            self.update_status("Error: 3D Model pipeline not initialized")
            messagebox.showerror(
                "Shape → 3D Error",
                "The 3D Model pipeline is not initialized.\n"
                "Please check the application logs for details."
            )
            return
            
        # Check if canvas is empty
        if self._check_canvas_empty():
            self.update_status("No shape detected. Please draw a shape first.")
            messagebox.showinfo(
                "Empty Canvas",
                "No shape detected.\n\n"
                "Please draw a geometric shape on the canvas first, "
                "then click 'Shape → 3D' again."
            )
            return
            
        # Disable the button during processing
        self.btn_shape_3d.configure(state="disabled", text="⏳ Processing...")
        self.update_status("Analyzing shape...")
        self.root.update_idletasks()

        # Tk widgets must only be read on the UI thread.  The exported PNG is
        # immutable, so only the expensive recognition/mesh/export work goes
        # to the worker.
        image_path = self.recognizer.capture_canvas() if self.recognizer else None
        if not image_path:
            self._handle_shape_3d_error("Failed to capture canvas")
            return
        
        # Run processing in a background thread to avoid freezing the GUI
        thread = threading.Thread(target=self._process_shape_3d, args=(image_path,), daemon=True)
        thread.start()
        
    def _process_shape_3d(self, image_path: Path) -> None:
        """
        Background thread function for Shape → 3D processing.
        This uses the ModelPipeline which internally uses ShapeRecognizer,
        GeometryGenerator, and STLExporter.
        """
        try:
            # The image was captured on the UI thread before this worker began.
            # STEP 1: Update status on main thread
            self.root.after(0, lambda: self.update_status("Recognizing shape..."))
            
            # STEP 2: Run the pipeline
            pipeline_result = self.model_pipeline.process(image_path)
            
            # STEP 3: Handle the result on the main thread
            self.root.after(0, lambda: self._handle_shape_3d_result(pipeline_result))
            
        except Exception as e:
            error_msg = f"Shape → 3D processing failed: {str(e)}"
            self.root.after(0, lambda: self._handle_shape_3d_error(error_msg))
            
    def _handle_shape_3d_result(self, pipeline_result) -> None:
        """
        Handle the result from the ModelPipeline for Shape → 3D.
        The pipeline returns a serializable dictionary plus the runtime mesh.
        
        Args:
            pipeline_result: Result dictionary from ModelPipeline.
        """
        # Re-enable the button
        self.btn_shape_3d.configure(state="normal", text="📐 Shape → 3D")
        
        # ModelPipeline returns a result dictionary, whereas the legacy code
        # below expected a RecognitionResult dataclass. Handle the dictionary
        # contract here and return before that obsolete branch.
        if isinstance(pipeline_result, dict):
            recognition = pipeline_result.get("recognition") or {}
            shape_name = str(recognition.get("shape", "unknown"))
            confidence = float(recognition.get("confidence", 0.0))
            if not pipeline_result.get("success", False):
                stage = pipeline_result.get("stage", "unknown")
                error = pipeline_result.get("error") or "3D model generation did not complete"
                self.update_status(f"3D {stage} failed: {error}")
                messagebox.showerror("Shape to 3D Failed", f"Stage: {stage}\n\n{error}")
                return
            stl_path = (pipeline_result.get("stl") or {}).get("path", "Unknown")
            mesh = pipeline_result.get("mesh")
            if self.mesh_viewer is None:
                self._handle_shape_3d_error("3D viewer is not initialized")
                return
            try:
                self.mesh_viewer.show_mesh(mesh, f"{shape_name.title()} 3D model")
            except Exception as exc:
                self._handle_shape_3d_error(f"Mesh generated but preview failed: {exc}")
                return
            self.update_status(f"3D model generated: {shape_name} ({confidence:.1%})")
            messagebox.showinfo(
                "3D Model Generated Successfully",
                f"3D Model Generated!\n\nShape: {shape_name}\nConfidence: {confidence:.1%}\nSTL File: {stl_path}"
            )
            self.ai_placeholder.configure(
                text=f"3D Model Generated\n\nShape: {shape_name}\nConfidence: {confidence:.1%}\nSTL: {Path(stl_path).name}"
            )
            return

        # Check if pipeline returned an error
        if hasattr(pipeline_result, 'error') and pipeline_result.error:
            self.update_status(f"❌ Recognition error: {pipeline_result.error}")
            messagebox.showerror(
                "Shape Recognition Failed",
                f"Shape recognition failed:\n\n{pipeline_result.error}\n\n"
                "Please draw a clearer geometric shape and try again."
            )
            return
        
        # Get shape and confidence from the RecognitionResult object
        shape_obj = getattr(pipeline_result, 'shape', None)
        confidence = getattr(pipeline_result, 'confidence', 0.0)
        
        # Get shape name
        if shape_obj is not None:
            if hasattr(shape_obj, 'value'):
                shape_name = shape_obj.value
            else:
                shape_name = str(shape_obj)
        else:
            shape_name = "Unknown"
        
        # Check if shape is UNKNOWN
        if shape_name.lower() == "unknown":
            self.update_status(f"❌ Could not confidently identify a supported shape.")
            messagebox.showwarning(
                "Shape Not Recognized",
                f"Could not confidently identify a supported shape.\n\n"
                f"Detected: {shape_name}\n"
                f"Confidence: {confidence:.1%}\n\n"
                "Please draw a clear geometric shape (circle, square, rectangle, triangle, ellipse, or line)."
            )
            return
        
        # Check confidence against 70% threshold
        if confidence >= 0.70:
            # Success - show details
            self.update_status(f"✅ 3D model generated! Shape: {shape_name} (Confidence: {confidence:.1%})")
            
            # Get STL path from pipeline result if available
            stl_path = "Unknown"
            if hasattr(pipeline_result, 'stl_path'):
                stl_path = pipeline_result.stl_path
            elif hasattr(pipeline_result, 'stl'):
                stl_path = getattr(pipeline_result.stl, 'path', "Unknown")
            
            # Show success message
            messagebox.showinfo(
                "3D Model Generated Successfully",
                f"🎉 3D Model Generated!\n\n"
                f"Shape: {shape_name}\n"
                f"Confidence: {confidence:.1%}\n"
                f"STL File: {stl_path}\n\n"
                f"The STL file has been saved and is ready for 3D printing."
            )
            
            # Update the AI placeholder with 3D info
            self.ai_placeholder.configure(
                text=f"✅ 3D Model Generated\n\n"
                     f"Shape: {shape_name}\n"
                     f"Confidence: {confidence:.1%}\n"
                     f"STL: {Path(stl_path).name if stl_path != 'Unknown' else 'Unknown'}\n\n"
                     f"The model is ready for 3D printing!"
            )
        else:
            # Confidence below 70% - do NOT generate geometry
            self.update_status(f"❌ Shape confidence too low: {confidence:.1%} (minimum 70%)")
            
            messagebox.showwarning(
                "Shape Confidence Too Low",
                f"Shape confidence is too low to generate a 3D model.\n\n"
                f"Detected shape: {shape_name}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Minimum required: 70%\n\n"
                f"Please draw a clearer shape and try again."
            )
            
    def _handle_shape_3d_error(self, error_msg: str) -> None:
        """
        Handle errors during Shape → 3D processing.
        
        Args:
            error_msg: Error message to display
        """
        # Re-enable the button
        self.btn_shape_3d.configure(state="normal", text="📐 Shape → 3D")
        
        self.update_status(f"❌ Error: {error_msg}")
        
        messagebox.showerror(
            "Shape → 3D Error",
            f"An error occurred during shape processing:\n\n"
            f"{error_msg}\n\n"
            f"Please try again or check the logs for more details."
        )
        
    # ======================== Toolbar Callbacks ========================

    def _toggle_shape_correction(self) -> None:
        """Enable or disable visual regularisation without disabling recognition."""
        enabled = not self.shape_correction_enabled.get()
        self.shape_correction_enabled.set(enabled)
        self.btn_shape_correct.configure(text=f"Shape Correct: {'ON' if enabled else 'OFF'}")
        self.update_status(f"Shape correction {'enabled' if enabled else 'disabled'}")

    def _on_stroke_finished(self, stroke_count: int) -> None:
        """Debounce adjacent strokes so words and multi-stroke letters stay together."""
        if not self.canvas or not self.automatic_recognizer:
            return
        self._logger.debug("[SHAPE] Stroke finished (count=%d)", stroke_count)
        if self._pending_group_start is None:
            self._pending_group_start = stroke_count - 1
        if self._recognition_after_id is not None:
            self.root.after_cancel(self._recognition_after_id)
        self._recognition_after_id = self.root.after(self.AUTO_RECOGNITION_DELAY_MS, self._start_automatic_recognition)

    def _group_image(self, start_index: int):
        """Render only the newly completed stroke group for independent recognition."""
        assert self.canvas is not None
        strokes = self.canvas.get_stroke_history()[start_index:]
        points = [point for stroke in strokes for point in stroke.points]
        if not points:
            return None
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        padding = 18
        width, height = max(80, int(max_x - min_x) + 2 * padding), max(80, int(max_y - min_y) + 2 * padding)
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            translated = [(x - min_x + padding, y - min_y + padding) for x, y in stroke.points]
            if len(translated) == 1:
                x, y = translated[0]
                radius = max(1, round(stroke.width)) / 2
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=stroke.color)
            else:
                draw.line(translated, fill=stroke.color, width=max(1, round(stroke.width)), joint="curve")
        import numpy as np
        return np.asarray(image), (min_x - padding, min_y - padding)

    def _start_automatic_recognition(self) -> None:
        self._recognition_after_id = None
        start_index = self._pending_group_start
        self._pending_group_start = None
        if start_index is None or not self.automatic_recognizer:
            return
        group_render = self._group_image(start_index)
        if group_render is None:
            return
        image, origin = group_render
        self.update_status("Recognizing drawing...")
        group_end = self.canvas.get_stroke_count() if self.canvas else start_index
        threading.Thread(
            target=self._process_automatic_recognition,
            args=(start_index, group_end, image, origin), daemon=True,
        ).start()

    def _process_automatic_recognition(self, start_index: int, group_end: int, image, origin) -> None:
        try:
            result = self.automatic_recognizer.analyze(
                image, shape_correction_enabled=self.shape_correction_enabled.get()
            ) if self.automatic_recognizer else None
            if result is not None:
                self.root.after(0, lambda: self._handle_automatic_recognition(start_index, group_end, result, origin))
        except Exception as exc:
            if __debug__:
                print(f"Automatic recognition failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _handle_automatic_recognition(
        self, start_index: int, group_end: int, result: UnifiedRecognitionResult, origin=(0, 0)
    ) -> None:
        """Apply automatic results on Tk's thread; result is an object, not a dict."""
        correction = False
        self._last_automatic_result = result
        self._logger.debug(
            "[SHAPE] Candidate=%s confidence=%.2f threshold=%.2f Shape Correct=%s",
            result.detected, result.confidence, self.SHAPE_CORRECTION_THRESHOLD,
            self.shape_correction_enabled.get(),
        )
        if (result.kind == "shape" and result.shape_result is not None
                and self.shape_correction_enabled.get()
                and result.confidence >= self.SHAPE_CORRECTION_THRESHOLD and self.canvas):
            features = result.shape_result.features
            x, y, width, height = features.bounding_rect
            features.bounding_rect = (int(x + origin[0]), int(y + origin[1]), width, height)
            features.center = (features.center[0] + origin[0], features.center[1] + origin[1])
            vertices = [(x + origin[0], y + origin[1]) for x, y in result.shape_result.raw_vertices]
            contour_points = [(x + origin[0], y + origin[1]) for x, y in result.shape_result.contour_points]
            self._logger.debug("[SHAPE] Replacing original strokes %d:%d", start_index, group_end)
            correction = self.canvas.replace_strokes_with_shape(
                start_index, group_end, result.shape_result.shape, features, vertices, contour_points
            )
            self._logger.debug("[SHAPE] Corrected shape inserted; canvas redraw requested=%s", correction)
        elif result.kind != "shape":
            self._logger.debug("[SHAPE] Correction skipped: text/character/unknown result")
        elif result.confidence < self.SHAPE_CORRECTION_THRESHOLD:
            self._logger.debug("[SHAPE] Correction skipped: confidence below threshold")
        elif not self.shape_correction_enabled.get():
            self._logger.debug("[SHAPE] Correction skipped: Shape Correct is OFF")
        if result.kind in {"character", "text"}:
            self._update_ocr_text(result.text)
        elif result.kind == "math":
            self._update_ocr_text(result.text)
            self._update_category("MATHEMATICS", result.confidence)
        elif result.kind == "shape":
            self._update_ocr_text(f"Shape: {result.detected} ({result.confidence:.0%})")
        self.update_status(
            f"Recognition: {result.kind.title()} / {result.detected} / {result.confidence:.0%} "
            f"/ correction: {'YES' if correction else 'NO'} / {result.model}"
        )
        self.board_objects.append(result.to_board_object((start_index, group_end), corrected=correction))
    
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
            self.update_status("Eraser selected")
        
    def _on_clear(self) -> None:
        """Handle clear canvas action."""
        if self.canvas:
            self.canvas.clear_canvas()
            self._update_ocr_text("No text recognized yet.\nDraw something and click 'Recognize'.")
            self._update_category("UNKNOWN", 0.0)
            self.update_status("Canvas cleared.")
        
    def _on_undo(self) -> None:
        """Handle undo action."""
        if self.canvas:
            if self.canvas.undo_last_stroke():
                self.update_status("Undo Successful")
            else:
                self.update_status("Nothing to Undo")
        
    def _on_redo(self) -> None:
        """Restore the most recently undone stroke or automatic correction."""
        if self.canvas and self.canvas.redo_last_stroke():
            self.update_status("Redo Successful")
        else:
            self.update_status("Nothing to Redo")
        
    def _on_save(self) -> None:
        """Handle save action (placeholder)."""
        self.update_status("Saving...")
        
    def _on_recognize(self) -> None:
        """Run OCR in a worker so Tesseract does not freeze Tk."""
        if self._check_canvas_empty():
            self._update_ocr_text("Empty canvas: draw text before recognition.")
            self.update_status("OCR skipped: canvas is empty")
            return
        if not self.recognizer or not self.ocr_processor:
            self._update_ocr_text("[OCR Error] OCR components are not initialized.")
            return
        self.btn_recognize.configure(state="disabled", text="Recognizing...")
        self.update_status("Capturing drawing and running OCR...")
        image_path = self.recognizer.capture_canvas()
        if not image_path:
            self._handle_ocr_result({
                "text": "[OCR Error] Canvas capture failed", "confidence": 0.0, "mode": "error"
            }, None)
            return
        threading.Thread(target=self._process_ocr, args=(image_path,), daemon=True).start()

    def _process_ocr(self, image_path: Path) -> None:
        """Worker half of the canvas -> OCR -> classifier flow."""
        try:
            ocr_result = self.ocr_processor.extract_text_with_confidence(image_path)
            text = ocr_result.get("text", "")
            classification = None
            if text and not text.startswith("[OCR Error]") and text != "No text recognized" and self.classifier:
                classification = self.classifier.classify(text)
            self.root.after(0, lambda: self._handle_ocr_result(ocr_result, classification))
        except Exception as exc:
            self.root.after(0, lambda: self._handle_ocr_result({
                "text": f"[OCR Error] {type(exc).__name__}: {exc}", "confidence": 0.0, "mode": "error"
            }, None))

    def _handle_ocr_result(self, ocr_result, classification) -> None:
        """Apply OCR results on Tk's main thread."""
        self.btn_recognize.configure(state="normal", text="Recognize")
        text = ocr_result.get("text", "")
        self._update_ocr_text(text)
        if classification:
            category = classification.get("category", "UNKNOWN")
            confidence = float(classification.get("confidence", 0.0))
            self._update_category(category, confidence)
            self.update_status(f"OCR complete ({ocr_result.get('mode', 'text')}): {category}")
        else:
            self._update_category("UNKNOWN", 0.0)
            self.update_status(f"OCR complete: {ocr_result.get('mode', 'unknown')}")
        
    def _on_ai(self) -> None:
        """Request a student-facing explanation only when the user asks for it."""
        if not self.ai_provider or not self.ai_provider.is_available:
            self.ai_placeholder.configure(
                text="NVIDIA AI is not configured. Set NVIDIA_API_KEY and NVIDIA_MODEL, then restart the app."
            )
            self.update_status("AI explanation unavailable: NVIDIA configuration missing")
            return
        self.btn_ai.configure(state="disabled", text="Explaining...")
        self.update_status("Requesting NVIDIA AI explanation...")
        threading.Thread(target=self._process_ai_explanation, args=(self._build_board_context(),), daemon=True).start()

    def _on_graph(self) -> None:
        """Plot the recognized equation locally in a small non-blocking Tk window."""
        text = self.ocr_text_box.get("1.0", tk.END).strip()
        math = self.math_engine.analyze(text, 1.0)
        if not math.parsed:
            self.update_status("Graph unavailable: recognize a valid equation first")
            messagebox.showinfo("Graph", "Recognize a valid mathematical expression before plotting it.")
            return
        try:
            graph = self.graph_engine.build(math.normalized)
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            graph_window = ctk.CTkToplevel(self.root)
            graph_window.title(f"Graph: {math.expression}")
            graph_window.geometry("560x460")
            figure_canvas = FigureCanvasTkAgg(self.graph_engine.figure(graph), master=graph_window)
            figure_canvas.draw()
            figure_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.board_objects.append(BoardObject("graph", math.expression, 1.0, metadata={"graph": graph.kind}))
            self.update_status(f"Graph created: {math.expression}")
        except Exception as exc:
            self.update_status("Graph generation failed")
            if __debug__:
                self._logger.debug("Graph generation failed: %s", type(exc).__name__)
            messagebox.showerror("Graph", "The recognized expression could not be plotted.")

    def _build_board_context(self) -> str:
        """Send recognized context rather than raw strokes to an optional remote provider."""
        text = self.ocr_text_box.get("1.0", tk.END).strip()
        context = ["Smart Blackboard context:", f"Recognized text: {text or 'none'}"]
        if self.board_objects:
            context.append("Board objects:\n" + "\n".join(obj.describe() for obj in self.board_objects[-20:]))
        if self._last_automatic_result:
            context.append(
                f"Latest recognition: {self._last_automatic_result.kind} = "
                f"{self._last_automatic_result.detected} "
                f"({self._last_automatic_result.confidence:.0%})"
            )
        if self.canvas:
            context.append(f"Board stroke objects: {self.canvas.get_stroke_count()}")
        return "\n".join(context)

    def _process_ai_explanation(self, context: str) -> None:
        try:
            assert self.ai_provider is not None
            explanation = self.ai_provider.explain(context)
            self.root.after(0, lambda: self._handle_ai_explanation(explanation, None))
        except AIProviderError as exc:
            self.root.after(0, lambda: self._handle_ai_explanation(None, str(exc)))
        except Exception:
            self.root.after(0, lambda: self._handle_ai_explanation(None, "NVIDIA AI request failed unexpectedly."))

    def _handle_ai_explanation(self, explanation: Optional[str], error: Optional[str]) -> None:
        self.btn_ai.configure(state="normal", text="🤖 AI Explain")
        if error:
            self.ai_placeholder.configure(text=error)
            self.update_status(error)
            return
        self.ai_placeholder.configure(text=explanation or "No explanation returned.")
        self.update_status("NVIDIA AI explanation complete")
        
    # ======================== Menu Callbacks ========================
    
    def _on_new(self) -> None:
        """Handle new file action."""
        if self.canvas:
            self.canvas.clear_canvas()
            self.board_objects.clear()
            self._update_ocr_text("No text recognized yet.\nDraw something and click 'Recognize'.")
            self._update_category("UNKNOWN", 0.0)
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
            "Features:\n"
            "✓ Drawing Canvas\n"
            "✓ OCR with Tesseract\n"
            "✓ Content Classification\n"
            "✓ Shape → 3D Model Generation\n"
            "✓ AI Ready Architecture\n\n"
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
        
    def get_ocr_processor(self) -> Optional[OCRProcessor]:
        """
        Get the OCR processor instance.
        
        Returns:
            Optional[OCRProcessor]: The OCR processor instance or None
        """
        return self.ocr_processor
        
    def get_classifier(self) -> Optional[ContentClassifier]:
        """
        Get the classifier instance.
        
        Returns:
            Optional[ContentClassifier]: The classifier instance or None
        """
        return self.classifier
        
    def get_model_pipeline(self) -> Optional[ModelPipeline]:
        """
        Get the ModelPipeline instance.
        
        Returns:
            Optional[ModelPipeline]: The ModelPipeline instance or None
        """
        return self.model_pipeline


if __name__ == "__main__":
    # This allows testing the window directly
    app = SmartBlackboard()
    app.run()

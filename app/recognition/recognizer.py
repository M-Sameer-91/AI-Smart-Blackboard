"""
recognizer.py - Canvas Capture Module for AI Smart Blackboard

This module provides the Recognizer class responsible for capturing
the current state of the drawing canvas and saving it as an image file.
It serves as the foundation for the AI pipeline by providing image data
to future OCR, recognition, and analysis modules.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageGrab
import tkinter as tk


class Recognizer:
    """
    Canvas capture and image saving utility for the AI Smart Blackboard.
    
    This class handles the capture of drawing canvas content and saves
    it as a PNG image with timestamp-based filenames. It is designed
    to be the first step in the AI pipeline, providing image data to
    future modules like OCR, math recognition, and AI analysis.
    
    Attributes:
        canvas: Reference to the DrawingCanvas instance
        output_dir: Path to the directory where images will be saved
        last_saved_path: Path of the most recently saved image
    """
    
    def __init__(self, canvas) -> None:
        """
        Initialize the Recognizer with a canvas instance.
        
        Args:
            canvas: The DrawingCanvas instance to capture from
            
        Raises:
            ValueError: If the provided canvas is invalid
        """
        if canvas is None:
            raise ValueError("Canvas instance cannot be None")
        
        self._canvas = canvas
        self._output_dir = Path("app/data/captured")
        self._last_saved_path: Optional[Path] = None
        
        # Ensure output directory exists
        self._create_output_directory()
        
    def _create_output_directory(self) -> None:
        """
        Create the output directory for captured images if it doesn't exist.
        
        Raises:
            PermissionError: If unable to create the directory due to permissions
            OSError: If directory creation fails for other reasons
        """
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied: Cannot create directory '{self._output_dir}'. "
                f"Please check file permissions."
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to create output directory '{self._output_dir}'. "
                f"Error: {str(e)}"
            ) from e
            
    def generate_filename(self) -> str:
        """
        Generate a timestamp-based filename for the captured image.
        
        Format: board_YYYYMMDD_HHMMSS.png
        
        Returns:
            str: Generated filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"board_{timestamp}.png"
        
    def capture_canvas(self) -> Optional[Path]:
        """
        Capture the current canvas content and save it as an image.
        
        This method extracts the canvas widget's position and size,
        captures the screen region, and saves it to the output directory.
        
        Returns:
            Optional[Path]: Path to the saved image file, or None if capture fails
            
        Raises:
            RuntimeError: If canvas capture fails unexpectedly
        """
        try:
            # Get the canvas widget
            canvas_widget = self._canvas.get_canvas()
            
            # Ensure canvas widget exists
            if not canvas_widget:
                raise RuntimeError("Canvas widget not available")
            
            # Update widget to ensure correct geometry
            canvas_widget.update_idletasks()
            
            # Get canvas position and size
            x = canvas_widget.winfo_rootx()
            y = canvas_widget.winfo_rooty()
            width = canvas_widget.winfo_width()
            height = canvas_widget.winfo_height()
            
            # Validate canvas dimensions
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid canvas dimensions: {width}x{height}")
            
            # Capture the canvas region from screen
            bbox = (x, y, x + width, y + height)
            screenshot = ImageGrab.grab(bbox=bbox)
            
            # Generate filename and save
            filename = self.generate_filename()
            image_path = self._output_dir / filename
            
            # Save the image
            self.save_image(screenshot, image_path)
            
            # Store the path of the last saved image
            self._last_saved_path = image_path
            
            return image_path
            
        except RuntimeError as e:
            print(f"Runtime Error during canvas capture: {str(e)}", file=sys.stderr)
            return None
            
        except Exception as e:
            print(f"Unexpected error during canvas capture: {str(e)}", file=sys.stderr)
            return None
            
    def save_image(self, image: Image.Image, path: Path) -> None:
        """
        Save a PIL Image to the specified path with error handling.
        
        Args:
            image: PIL Image object to save
            path: Path where the image should be saved
            
        Raises:
            ValueError: If image is None or path is invalid
            PermissionError: If unable to write to the directory
            OSError: If image saving fails
        """
        if image is None:
            raise ValueError("Image cannot be None")
            
        if path is None:
            raise ValueError("Path cannot be None")
            
        # Ensure the path has a .png extension
        if not path.suffix.lower() == '.png':
            path = path.with_suffix('.png')
            
        try:
            # Save the image as PNG
            image.save(path, 'PNG', optimize=True)
            
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied: Cannot write to '{path}'. "
                f"Please check file permissions."
            ) from e
            
        except OSError as e:
            raise OSError(
                f"Failed to save image to '{path}'. "
                f"Error: {str(e)}"
            ) from e
            
    def get_last_saved_path(self) -> Optional[Path]:
        """
        Get the path of the most recently saved image.
        
        Returns:
            Optional[Path]: Path to the last saved image, or None if no image has been saved
        """
        return self._last_saved_path
        
    def get_output_directory(self) -> Path:
        """
        Get the output directory path.
        
        Returns:
            Path: The directory where images are saved
        """
        return self._output_dir
        
    def get_image_count(self) -> int:
        """
        Get the number of images in the output directory.
        
        Returns:
            int: Count of PNG files in the output directory
        """
        try:
            if not self._output_dir.exists():
                return 0
            return len(list(self._output_dir.glob("*.png")))
        except Exception:
            return 0
            
    def clear_output_directory(self) -> bool:
        """
        Clear all images from the output directory.
        
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            This method is intended for development and testing purposes.
        """
        try:
            if not self._output_dir.exists():
                return True
                
            for file in self._output_dir.glob("*.png"):
                file.unlink()
            return True
            
        except Exception as e:
            print(f"Error clearing output directory: {str(e)}", file=sys.stderr)
            return False
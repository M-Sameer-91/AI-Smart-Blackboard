"""
shape_recognizer.py - 2D Shape Recognition Module for AI Smart Blackboard

This module provides computer vision-based shape recognition for drawings
on the Smart Blackboard. It identifies common 2D shapes including circles,
rectangles, squares, triangles, ellipses, and lines from imperfect hand-drawn
input. The module is designed for future extension to support 3D geometry
generation and STL export.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
from enum import Enum
from dataclasses import dataclass, field
import time
import sys

from app.config import RECOGNITION_SETTINGS


class ShapeType(Enum):
    """Enumeration of supported shape types."""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    ELLIPSE = "ellipse"
    LINE = "line"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    HEPTAGON = "heptagon"
    OCTAGON = "octagon"
    NONAGON = "nonagon"
    DECAGON = "decagon"
    UNDECAGON = "undecagon"
    DODECAGON = "dodecagon"
    TRAPEZOID = "trapezoid"
    PARALLELOGRAM = "parallelogram"
    RHOMBUS = "rhombus"
    KITE = "kite"
    STAR = "star"
    ARROW = "arrow"
    CROSS = "cross"
    HEART = "heart"
    SEMICIRCLE = "semicircle"
    QUARTER_CIRCLE = "quarter_circle"
    CRESCENT = "crescent"
    UNKNOWN = "unknown"


@dataclass
class ShapeFeatures:
    """Container for geometric features extracted from a contour."""
    area: float = 0.0
    perimeter: float = 0.0
    vertices: int = 0
    width: float = 0.0
    height: float = 0.0
    aspect_ratio: float = 0.0
    circularity: float = 0.0
    extent: float = 0.0
    convexity: float = 0.0
    solidity: float = 0.0
    bounding_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[float, float] = (0.0, 0.0)
    contour_area_ratio: float = 0.0
    elongation: float = 0.0
    rotated_width: float = 0.0
    rotated_height: float = 0.0
    rotated_aspect_ratio: float = 0.0
    rectangularity: float = 0.0
    fitted_ellipse_area: float = 0.0
    ellipse_fit_error: float = 0.0
    internal_angles: List[float] = field(default_factory=list)
    angle_score: float = 0.0
    side_regularity: float = 0.0
    radius_consistency: float = 0.0
    closure: float = 0.0


@dataclass
class RecognitionResult:
    """Container for shape recognition results."""
    shape: ShapeType
    confidence: float
    features: ShapeFeatures
    raw_vertices: List[Tuple[float, float]] = field(default_factory=list)
    contour_points: List[Tuple[float, float]] = field(default_factory=list)
    error: Optional[str] = None


class ShapeRecognizer:
    """
    Advanced shape recognizer for hand-drawn 2D shapes.
    
    This class implements a robust computer vision pipeline to recognize
    hand-drawn shapes from the Smart Blackboard. It uses contour analysis
    and geometric feature extraction to classify shapes with confidence scores.
    
    The recognition pipeline includes:
    1. Image preprocessing (grayscale, thresholding, morphological cleanup)
    2. Contour detection and selection
    3. Geometric feature extraction
    4. Shape classification based on features
    5. Confidence calculation
    
    Designed to tolerate imperfect drawings from mouse or stylus input.
    """
    
    def __init__(
        self,
        min_contour_area: int = 200,
        max_contour_area: Optional[int] = None,
        epsilon_factor: float = 0.02,
        circularity_threshold: float = 0.6,
        aspect_ratio_tolerance: float = 0.18,
        angle_tolerance: float = RECOGNITION_SETTINGS.polygon_angle_tolerance,
        line_thickness_threshold: float = 0.1,
        debug: bool = True
    ) -> None:
        """
        Initialize the ShapeRecognizer with configurable parameters.
        
        Args:
            min_contour_area: Minimum contour area to consider (default: 200)
            max_contour_area: Maximum contour area (default: None)
            epsilon_factor: Factor for contour approximation (default: 0.02)
            circularity_threshold: Threshold for circularity detection (default: 0.6)
            aspect_ratio_tolerance: Tolerance for aspect ratio comparisons (default: 0.18)
            angle_tolerance: Tolerance for angle comparisons in degrees (default: 15.0)
            line_thickness_threshold: Threshold for line detection (default: 0.1)
            debug: Enable debug logging (default: True)
        """
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.epsilon_factor = epsilon_factor
        self.circularity_threshold = circularity_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.angle_tolerance = angle_tolerance
        self.line_thickness_threshold = line_thickness_threshold
        self.debug = debug
        self._debug_images_dir = Path("app/data/debug_shapes")
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self._logger = logging.getLogger(__name__)
        if debug:
            self._logger.setLevel(logging.DEBUG)
            self._debug_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Morphological kernels
        self._morph_kernel = np.ones((3, 3), np.uint8)
        self._close_kernel = np.ones((5, 5), np.uint8)
        self._large_close_kernel = np.ones((7, 7), np.uint8)
        
    def _save_debug_image(self, image: np.ndarray, name: str) -> None:
        """Save debug image if debug mode is enabled."""
        if self.debug:
            timestamp = int(time.time() * 1000)
            path = self._debug_images_dir / f"{name}_{timestamp}.png"
            cv2.imwrite(str(path), image)
            self._logger.debug(f"Saved debug image: {path}")
    
    def _print_debug_header(self, image: Union[str, Path, np.ndarray, Image.Image]) -> None:
        """Print comprehensive debug information about the input image."""
        print("\n" + "=" * 70)
        print("=== SHAPE RECOGNIZER DEBUG ===")
        print("=" * 70)
        
        # Image path info
        if isinstance(image, (str, Path)):
            print(f"\nImage: {image}")
            print(f"Exists: {Path(image).exists()}")
            if Path(image).exists():
                print(f"Size: {Path(image).stat().st_size} bytes")
        
        # Load image for analysis
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                print(f"ERROR: Failed to load image: {image}")
                return
        elif isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            print(f"ERROR: Unsupported image type: {type(image)}")
            return
        
        print(f"\nImage Shape: {img.shape}")
        print(f"Image Mode: {img.dtype}")
        
        # Grayscale analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"Channels: 3 (BGR)")
        else:
            gray = img
            print(f"Channels: 1 (Grayscale)")
        
        print(f"\nGrayscale Analysis:")
        print(f"  Min: {np.min(gray)}")
        print(f"  Max: {np.max(gray)}")
        print(f"  Mean: {np.mean(gray):.2f}")
        print(f"  Std: {np.std(gray):.2f}")
        print(f"  Unique values: {len(np.unique(gray))}")
        
        # Detect if image is likely a blackboard capture
        self._detect_image_type(gray)
    
    def _detect_image_type(self, gray: np.ndarray) -> None:
        """Detect if image is likely a blackboard capture with drawing."""
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        print(f"\nImage Type Analysis:")
        if mean_intensity > 200 and std_intensity < 20:
            print("  ⚠️  Image appears to be mostly white/empty (bright, low variance)")
            print("     This may indicate a blank canvas or overexposed image.")
        elif mean_intensity < 50 and std_intensity < 20:
            print("  ⚠️  Image appears to be mostly black/empty (dark, low variance)")
            print("     This may indicate a blank canvas or underexposed image.")
        else:
            print("  ✅ Image has reasonable contrast and variance for shape detection.")
    
    def _print_preprocess_debug(self, binary: np.ndarray, fg_percentage: float) -> None:
        """Print debug information after preprocessing."""
        print(f"\nPreprocessing Result:")
        print(f"  Binary shape: {binary.shape}")
        print(f"  Foreground pixels: {np.count_nonzero(binary)}")
        print(f"  Total pixels: {binary.size}")
        print(f"  Foreground percentage: {fg_percentage:.2%}")
        
        if fg_percentage < 0.001:
            print("\n⚠️  WARNING: PREPROCESSING PRODUCED ALMOST NO FOREGROUND")
            print("   The image may be empty or the thresholding may have failed.")
            print("   Check if the drawing color matches the expected polarity.")
        elif fg_percentage > 0.4:
            print("\n⚠️  WARNING: PREPROCESSING MAY HAVE CAPTURED THE BACKGROUND")
            print("   The foreground percentage is very high.")
            print("   The thresholding may have inverted the image incorrectly.")
        elif 0.005 <= fg_percentage <= 0.10:
            print("  ✅ Good foreground percentage for shape detection.")
    
    def _print_contour_debug(self, contours: List[np.ndarray], image_shape: Tuple[int, int]) -> None:
        """Print debug information about detected contours."""
        print(f"\nContour Detection:")
        print(f"  Total contours found: {len(contours)}")
        
        if not contours:
            print("  ⚠️  No contours found!")
            return
        
        height, width = image_shape
        print(f"  Image dimensions: {width} x {height}")
        
        for i, contour in enumerate(contours[:10]):  # Show first 10
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            touches = self._contour_touches_border(contour, image_shape)
            print(f"\n  Contour {i}:")
            print(f"    Area: {area:.0f}")
            print(f"    Bounding Box: ({x}, {y}) - {w}x{h}")
            print(f"    Touches border: {touches}")
        
        if len(contours) > 10:
            print(f"\n  ... and {len(contours) - 10} more contours")
    
    def _contour_touches_border(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, bool]:
        """Check if a contour touches the image border."""
        height, width = image_shape
        x, y, w, h = cv2.boundingRect(contour)
        margin = 2
        
        return {
            "top": y <= margin,
            "bottom": y + h >= height - margin,
            "left": x <= margin,
            "right": x + w >= width - margin
        }
    
    def _print_features_debug(self, features: ShapeFeatures) -> None:
        """Print debug information about extracted features."""
        print(f"\nExtracted Features:")
        print(f"  Area: {features.area:.0f}")
        print(f"  Perimeter: {features.perimeter:.1f}")
        print(f"  Vertices: {features.vertices}")
        print(f"  Width x Height: {features.width:.0f} x {features.height:.0f}")
        print(f"  Aspect Ratio: {features.aspect_ratio:.3f}")
        print(f"  Rotated Aspect Ratio: {features.rotated_aspect_ratio:.3f}")
        print(f"  Circularity: {features.circularity:.3f}")
        print(f"  Solidity: {features.solidity:.3f}")
        print(f"  Extent: {features.extent:.3f}")
        print(f"  Elongation: {features.elongation:.1f}")
        print(f"  Convexity: {features.convexity:.3f}")
        print(f"  Contour Area Ratio: {features.contour_area_ratio:.3f}")
    
    def _print_result_debug(self, shape_type: ShapeType, confidence: float) -> None:
        """Print debug information about the final result."""
        print(f"\nFinal Result:")
        print(f"  Shape: {shape_type.value}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Confidence meets 70% threshold: {'✅ YES' if confidence >= 0.70 else '❌ NO'}")
        
        if shape_type == ShapeType.UNKNOWN and confidence == 0.0:
            print("\n⚠️  CRITICAL: Recognition returned UNKNOWN with 0.0 confidence")
            print("   This indicates the recognition pipeline failed completely.")
            print("   Check the preprocessing and contour detection steps above.")
        
        print("\n" + "=" * 70)
        
    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess the input image for shape recognition.
        
        Args:
            image: Input image as path, numpy array, or PIL Image
            
        Returns:
            np.ndarray: Preprocessed binary image
            
        Raises:
            ValueError: If image loading or preprocessing fails
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Save original grayscale for debug
        self._save_debug_image(gray, "01_grayscale")
        
        # Determine threshold method based on image characteristics
        mean_intensity = np.mean(gray)
        
        if self.debug:
            self._logger.debug(f"Mean intensity: {mean_intensity:.2f}")
        
        # Try both polarities and choose the one that produces better foreground
        # Polarity 1: Dark background, light drawing (THRESH_BINARY_INV)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_inv = np.count_nonzero(binary_inv) / binary_inv.size
        self._save_debug_image(binary_inv, "02_threshold_inv")
        
        # Polarity 2: Light background, dark drawing (THRESH_BINARY)
        _, binary_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_normal = np.count_nonzero(binary_normal) / binary_normal.size
        self._save_debug_image(binary_normal, "03_threshold_normal")
        
        if self.debug:
            self._logger.debug(f"FG percentage (INV): {fg_inv:.2%}")
            self._logger.debug(f"FG percentage (Normal): {fg_normal:.2%}")
        
        # Choose the polarity that gives reasonable foreground percentage
        # Target: 0.5% to 30% foreground
        target_min = 0.005
        target_max = 0.30
        
        # Also check if one polarity gives a reasonable foreground while the other is clearly background
        if target_min <= fg_inv <= target_max and (fg_normal < 0.001 or fg_normal > 0.5):
            binary = binary_inv
            polarity = "INV (dark background)"
        elif target_min <= fg_normal <= target_max and (fg_inv < 0.001 or fg_inv > 0.5):
            binary = binary_normal
            polarity = "Normal (light background)"
        elif target_min <= fg_inv <= target_max:
            binary = binary_inv
            polarity = "INV (dark background)"
        elif target_min <= fg_normal <= target_max:
            binary = binary_normal
            polarity = "Normal (light background)"
        else:
            # Both are outside range - use the one with more reasonable foreground
            # Prefer the one with more foreground (likely the drawing)
            if fg_inv > fg_normal and fg_inv < 0.5:
                binary = binary_inv
                polarity = "INV (more foreground)"
            elif fg_normal > fg_inv and fg_normal < 0.5:
                binary = binary_normal
                polarity = "Normal (more foreground)"
            else:
                # If both are weird, default to INV (typical for blackboard)
                binary = binary_inv
                polarity = "INV (default)"
        
        if self.debug:
            self._logger.debug(f"Selected polarity: {polarity}")
        
        # If foreground is too low or too high, try adaptive thresholding
        fg_percentage = np.count_nonzero(binary) / binary.size
        if fg_percentage < 0.001 or fg_percentage > 0.5:
            if self.debug:
                self._logger.debug("Foreground outside target range, trying adaptive thresholding")
            if polarity.startswith("INV") or polarity.endswith("INV"):
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 15, 3
                )
            else:
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 3
                )
        
        # Save selected threshold
        self._save_debug_image(binary, "04_selected_threshold")
        
        # Print preprocessing diagnostics only when explicitly requested.
        fg_percentage = np.count_nonzero(binary) / binary.size
        if self.debug:
            self._print_preprocess_debug(binary, fg_percentage)
        
        # Morphological cleanup - close gaps first
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_kernel, iterations=2)
        
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
        
        # Dilate slightly to connect nearby components
        binary = cv2.dilate(binary, self._morph_kernel, iterations=1)
        
        self._save_debug_image(binary, "05_morphology")
        
        return binary
        
    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find and filter contours from the binary image.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            List[np.ndarray]: List of filtered contours
        """
        height, width = binary_image.shape
        image_area = height * width
        
        # Find contours using RETR_EXTERNAL to get outer boundaries only
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if self.debug:
            self._print_contour_debug(contours, binary_image.shape)
        
        if not contours:
            self._logger.warning("No contours found")
            return []
        
        if self.debug:
            self._logger.debug(f"Found {len(contours)} contours before filtering")
        
        # Calculate adaptive minimum area
        min_area = max(self.min_contour_area, image_area * 0.0005)  # 0.05% of image
        
        # Filter contours
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small
            if area < min_area:
                if self.debug:
                    self._logger.debug(f"Contour area {area:.0f} < min_area {min_area:.0f}, skipping")
                continue
            
            # Skip if too large (likely noise or border)
            if self.max_contour_area and area > self.max_contour_area:
                if self.debug:
                    self._logger.debug(f"Contour area {area:.0f} > max_area {self.max_contour_area}, skipping")
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if contour is likely background noise (very thin or very large)
            if w < 5 or h < 5:
                if self.debug:
                    self._logger.debug(f"Contour too thin: {w}x{h}, skipping")
                continue
            
            # Check if contour touches border
            touches_border = self._contour_touches_border(contour, binary_image.shape)
            
            # Skip if contour is likely the entire image border
            if all(touches_border.values()):
                if self.debug:
                    self._logger.debug("Contour touches all borders (likely background), skipping")
                continue
            
            # Also skip if the contour area is more than 90% of the image (likely background)
            if area > image_area * 0.9:
                if self.debug:
                    self._logger.debug(f"Contour area {area:.0f} > 90% of image, skipping (likely background)")
                continue
            
            # Allow contours that touch one or two borders (near edge drawings)
            filtered_contours.append(contour)
        
        if not filtered_contours:
            self._logger.warning(f"No contours passed filtering (min_area: {min_area:.0f})")
            return []
        
        # Sort by area (largest first)
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        if self.debug:
            self._logger.debug(f"Kept {len(filtered_contours)} contours after filtering")
            for i, c in enumerate(filtered_contours[:5]):
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                self._logger.debug(f"  Contour {i}: area={area:.0f}, bbox=({x},{y}) {w}x{h}")
        
        # If we have multiple contours, check if the largest is actually the background
        # and the second largest is the real shape
        if len(filtered_contours) > 1:
            largest_area = cv2.contourArea(filtered_contours[0])
            second_area = cv2.contourArea(filtered_contours[1])
            
            # If largest is more than 10x the second, it might be background
            if largest_area > second_area * 10:
                x, y, w, h = cv2.boundingRect(filtered_contours[0])
                # Check if the largest contour spans most of the image
                if w > width * 0.8 or h > height * 0.8:
                    if self.debug:
                        self._logger.debug(f"Largest contour may be background, considering second contour")
                    # Move the second contour to the front
                    filtered_contours.insert(0, filtered_contours.pop(1))
        
        return filtered_contours
        
    def extract_features(self, contour: np.ndarray) -> ShapeFeatures:
        """
        Extract geometric features from a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            ShapeFeatures: Extracted features
        """
        features = ShapeFeatures()
        
        # Basic properties
        features.area = cv2.contourArea(contour)
        features.perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        features.bounding_rect = (x, y, w, h)
        features.width = float(w)
        features.height = float(h)
        features.aspect_ratio = features.width / features.height if features.height > 0 else 0
        
        # Rotated bounding rectangle
        if len(contour) > 4:
            rect = cv2.minAreaRect(contour)
            (rx, ry), (rw, rh), _ = rect
            features.rotated_width = rw
            features.rotated_height = rh
            features.rotated_aspect_ratio = rw / rh if rh > 0 else 0
        else:
            features.rotated_width = features.width
            features.rotated_height = features.height
            features.rotated_aspect_ratio = features.aspect_ratio
        
        # Center
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            features.center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        else:
            features.center = (x + w / 2, y + h / 2)
        
        # Circularity
        if features.perimeter > 0:
            features.circularity = (4 * np.pi * features.area) / (features.perimeter ** 2)
        else:
            features.circularity = 0.0
        
        # Extent (area / bounding rectangle area)
        bounding_area = features.width * features.height
        features.extent = features.area / bounding_area if bounding_area > 0 else 0.0
        
        # Convexity (convex hull area / contour area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features.convexity = hull_area / features.area if features.area > 0 else 0.0
        
        # Solidity (area / convex hull area)
        features.solidity = features.area / hull_area if hull_area > 0 else 0.0
        
        # Elongation
        if features.width > features.height:
            features.elongation = features.width / (features.height + 1)
        else:
            features.elongation = features.height / (features.width + 1)
        
        # Contour area ratio
        features.contour_area_ratio = features.area / (features.width * features.height) if (features.width * features.height) > 0 else 0.0
        
        # Rectangularity
        features.rectangularity = features.area / (features.width * features.height) if (features.width * features.height) > 0 else 0.0
        
        # Approximate contour
        epsilon = self.epsilon_factor * features.perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        features.vertices = len(approx)
        vertices = [(float(p[0][0]), float(p[0][1])) for p in approx]
        features.internal_angles = self._internal_angles(vertices)
        if len(vertices) >= 3:
            sides = self._side_lengths(vertices)
            features.side_regularity = max(0.0, 1.0 - float(np.std(sides) / max(np.mean(sides), 1.0)))
        # Radial consistency is valuable circle evidence and is deliberately
        # separate from contour circularity (which can also be high for glyphs).
        points = contour.reshape((-1, 2)).astype(float)
        distances = np.linalg.norm(points - np.asarray(features.center), axis=1)
        if len(distances) and float(np.mean(distances)) > 0:
            features.radius_consistency = max(0.0, 1.0 - float(np.std(distances) / np.mean(distances)))
        features.closure = 1.0 if len(contour) >= 3 else 0.0
        
        if self.debug:
            self._logger.debug(f"Features: area={features.area:.0f}, perimeter={features.perimeter:.1f}, "
                              f"vertices={features.vertices}, aspect_ratio={features.aspect_ratio:.3f}, "
                              f"circularity={features.circularity:.3f}, solidity={features.solidity:.3f}")
        
        return features
        
    def recognize_shape(self, contour: np.ndarray) -> Tuple[ShapeType, float, ShapeFeatures, List[Tuple[float, float]]]:
        """
        Recognize the shape from a contour.
        
        Args:
            contour: Input contour
            
        Returns:
            Tuple[ShapeType, float, ShapeFeatures, List]: (shape_type, confidence, features, vertices)
        """
        # Extract features
        features = self.extract_features(contour)
        
        # Get approximated vertices
        epsilon = self.epsilon_factor * features.perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = [(float(p[0][0]), float(p[0][1])) for p in approx]
        
        # Determine shape type based on features
        shape_type, confidence = self._classify_shape(features, contour, vertices)
        
        return shape_type, confidence, features, vertices
        
    def _classify_shape(
        self,
        features: ShapeFeatures,
        contour: np.ndarray,
        vertices: List[Tuple[float, float]]
    ) -> Tuple[ShapeType, float]:
        """
        Classify the shape based on geometric features.
        
        Args:
            features: Extracted features
            contour: Original contour
            vertices: Approximated vertices
            
        Returns:
            Tuple[ShapeType, float]: (shape_type, confidence)
        """
        # Check for empty or invalid contour
        if features.area < 100 or features.perimeter < 10:
            return ShapeType.UNKNOWN, 0.1
        
        # Check for line
        line_result = self._is_line(features, contour)
        if line_result:
            shape_type, confidence = line_result
            return shape_type, confidence
        
        # Check for triangle (before rectangle because triangles can have 3 vertices)
        tri_result = self._is_triangle(features, vertices)
        if tri_result:
            shape_type, confidence = tri_result
            return shape_type, confidence
        
        quadrilateral_result = self._is_quadrilateral(features, vertices)
        if quadrilateral_result:
            return quadrilateral_result

        # Check for rectangle/square after non-rectangular quadrilaterals.
        rect_result = self._is_rectangle(features, vertices, contour)
        if rect_result:
            return rect_result

        polygon_result = self._is_regular_polygon(features, vertices)
        if polygon_result:
            return polygon_result

        # Check for circle
        circle_result = self._is_circle(features, contour)
        if circle_result:
            return circle_result
        
        # Check for ellipse
        ellipse_result = self._is_ellipse(features, contour)
        if ellipse_result:
            return ellipse_result

        symbolic_result = self._is_symbolic_shape(features, vertices)
        if symbolic_result:
            return symbolic_result
        
        # If it's somewhat round but not meeting criteria, check again
        if features.circularity > 0.5 and features.aspect_ratio > 0.5 and features.aspect_ratio < 1.5:
            return ShapeType.CIRCLE, 0.6
        
        # Unknown shape
        confidence = max(0.1, min(0.4, features.circularity * 0.4))
        return ShapeType.UNKNOWN, confidence

    @staticmethod
    def _side_lengths(vertices: List[Tuple[float, float]]) -> np.ndarray:
        points = np.asarray(vertices, dtype=float)
        return np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)

    @staticmethod
    def _internal_angles(vertices: List[Tuple[float, float]]) -> List[float]:
        """Return stable internal angles for a closed simplified contour."""
        if len(vertices) < 3:
            return []
        points = np.asarray(vertices, dtype=float)
        angles: List[float] = []
        for index, point in enumerate(points):
            before, after = points[index - 1] - point, points[(index + 1) % len(points)] - point
            denominator = np.linalg.norm(before) * np.linalg.norm(after)
            if denominator > 1e-6:
                angles.append(float(np.degrees(np.arccos(np.clip(np.dot(before, after) / denominator, -1, 1)))))
        return angles

    def _angle_similarity(self, vertices: List[Tuple[float, float]], expected: float) -> float:
        angles = self._internal_angles(vertices)
        if not angles:
            return 0.0
        mean_error = float(np.mean(np.abs(np.asarray(angles) - expected)))
        # The configured tolerance remains degree-based for compatibility.
        return max(0.0, 1.0 - mean_error / max(self.angle_tolerance * 2.0, 1.0))

    @staticmethod
    def _parallel(a: np.ndarray, b: np.ndarray, tolerance: float = 0.22) -> bool:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        cross_product = float(a[0] * b[1] - a[1] * b[0])
        return denominator > 0 and abs(cross_product / denominator) <= tolerance

    def _is_regular_polygon(self, features: ShapeFeatures, vertices: List[Tuple[float, float]]) -> Optional[Tuple[ShapeType, float]]:
        """Classify convex, approximately regular polygons without per-N code."""
        polygon_types = {
            5: ShapeType.PENTAGON, 6: ShapeType.HEXAGON, 7: ShapeType.HEPTAGON,
            8: ShapeType.OCTAGON, 9: ShapeType.NONAGON, 10: ShapeType.DECAGON,
            11: ShapeType.UNDECAGON, 12: ShapeType.DODECAGON,
        }
        count = len(vertices)
        if count not in polygon_types or features.solidity < 0.82:
            return None
        sides = self._side_lengths(vertices)
        variation = float(np.std(sides) / max(np.mean(sides), 1.0))
        # This narrow threshold keeps a smoothly sampled circle (which often
        # approximates to eight points) from becoming an octagon.
        expected_angle = (count - 2) * 180.0 / count
        angle_score = self._angle_similarity(vertices, expected_angle)
        features.angle_score = angle_score
        # Tolerate imperfect hand drawing, but require both side and angle
        # evidence so circles sampled into many points are not polygons.
        # A smooth circle frequently approximates to eight evenly-spaced
        # vertices.  Its high circularity is stronger evidence than the
        # polygon approximation; do not promote it to an octagon.
        if features.circularity >= 0.84 or variation > 0.16 or angle_score < 0.45:
            return None
        confidence = min(0.95, 0.68 + 0.16 * angle_score + 0.10 * (1.0 - variation / 0.16) + (0.04 if features.solidity > 0.92 else 0))
        return polygon_types[count], confidence

    def _is_quadrilateral(self, features: ShapeFeatures, vertices: List[Tuple[float, float]]) -> Optional[Tuple[ShapeType, float]]:
        """Recognize non-rectangular quadrilaterals from side and parallelism ratios."""
        if len(vertices) != 4 or features.solidity < 0.75:
            return None
        points = np.asarray(vertices, dtype=float)
        sides = self._side_lengths(vertices)
        vectors = np.roll(points, -1, axis=0) - points
        mean = max(float(np.mean(sides)), 1.0)
        equal = float(np.std(sides) / mean) < 0.22
        opposite_a = self._parallel(vectors[0], vectors[2])
        opposite_b = self._parallel(vectors[1], vectors[3])
        # Squares are rhombi mathematically, but retain the established square
        # classification when all four angles are also approximately right.
        angles = []
        for index in range(4):
            a, b, c = points[index - 1], points[index], points[(index + 1) % 4]
            first, second = a - b, c - b
            denom = np.linalg.norm(first) * np.linalg.norm(second)
            if denom:
                angles.append(np.degrees(np.arccos(np.clip(np.dot(first, second) / denom, -1, 1))))
        if equal and not all(abs(angle - 90) < self.angle_tolerance * 1.5 for angle in angles):
            return ShapeType.RHOMBUS, 0.82
        if opposite_a and opposite_b:
            if all(abs(angle - 90) < self.angle_tolerance * 1.5 for angle in angles):
                return None
            return ShapeType.PARALLELOGRAM, 0.80
        if opposite_a or opposite_b:
            return ShapeType.TRAPEZOID, 0.78
        # A kite has two unequal adjacent equal-side pairs.
        adjacent_pairs = abs(sides[0] - sides[1]) / mean < 0.24 and abs(sides[2] - sides[3]) / mean < 0.24
        if adjacent_pairs:
            return ShapeType.KITE, 0.76
        return None

    def _is_symbolic_shape(self, features: ShapeFeatures, vertices: List[Tuple[float, float]]) -> Optional[Tuple[ShapeType, float]]:
        """Small set of scalable feature rules for common concave board symbols."""
        count = len(vertices)
        # Concave outlines have a notably lower solidity than polygons/circles.
        if features.solidity < 0.78:
            if 8 <= count <= 14 and features.circularity < 0.55:
                return ShapeType.STAR, 0.75
            if 10 <= count <= 16 and 0.35 <= features.extent <= 0.70:
                return ShapeType.CROSS, 0.72
            if 5 <= count <= 9 and features.aspect_ratio > 1.25:
                return ShapeType.ARROW, 0.71
            if 6 <= count <= 14 and 0.55 <= features.aspect_ratio <= 1.35:
                return ShapeType.HEART, 0.70
            if features.circularity > 0.25:
                return ShapeType.CRESCENT, 0.70
        # A closed semicircle/quarter-circle has one or two straight chord edges
        # plus a curved arc, therefore lower circularity than a complete circle.
        if 0.45 <= features.circularity <= 0.80 and features.solidity > 0.85:
            if 0.80 <= features.aspect_ratio <= 1.25 and 3 <= count <= 7:
                return ShapeType.SEMICIRCLE, 0.70
            if 3 <= count <= 6:
                return ShapeType.QUARTER_CIRCLE, 0.70
        return None
        
    def _is_line(self, features: ShapeFeatures, contour: np.ndarray) -> Optional[Tuple[ShapeType, float]]:
        """
        Check if the contour represents a line.
        
        Args:
            features: Extracted features
            contour: Original contour
            
        Returns:
            Optional[Tuple[ShapeType, float]]: (LINE, confidence) or None
        """
        # Lines have very small area relative to bounding box
        area_ratio = features.contour_area_ratio
        
        # Lines are highly elongated
        elongation = features.elongation
        
        # Lines have very thin shape
        min_side = min(features.width, features.height)
        max_side = max(features.width, features.height)
        
        if min_side < 1:
            return None
            
        thinness = max_side / min_side
        
        # Check if it's a line
        # A thick drawn line can occupy most of its bounding rectangle, so
        # contour area ratio is not a reliable rejection criterion here.
        if elongation > 3.0 and thinness > 6.0:
            confidence = 0.75 + (0.1 * (thinness / 20.0))
            return ShapeType.LINE, min(confidence, 0.95)
            
        # Alternative line detection for very thin shapes
        if thinness > 12.0 and features.area < 3000:
            confidence = 0.80
            return ShapeType.LINE, confidence
            
        return None
        
    def _is_triangle(self, features: ShapeFeatures, vertices: List[Tuple[float, float]]) -> Optional[Tuple[ShapeType, float]]:
        """
        Check if the contour represents a triangle.
        
        Args:
            features: Extracted features
            vertices: Approximated vertices
            
        Returns:
            Optional[Tuple[ShapeType, float]]: (TRIANGLE, confidence) or None
        """
        if len(vertices) < 3:
            return None
            
        # Triangle should have 3-6 vertices (hand-drawn triangles often have more)
        if len(vertices) > 7:
            return None
            
        # Check if convexity is high (triangles should be convex)
        if features.convexity < 0.6:
            return None
            
        # Calculate angles
        angles = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            p3 = vertices[(i + 2) % len(vertices)]
            
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1 or norm2 < 1:
                continue
                
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))
        
        if len(angles) < 3:
            return None
            
        # Triangle angles should sum to ~180 degrees
        angle_sum = sum(angles)
        
        # Check if angles are reasonable for a triangle (20-150 degrees each)
        valid_angles = all(20 <= a <= 150 for a in angles)
        
        features.angle_score = self._angle_similarity(vertices, 60.0) if len(vertices) == 3 else max(0.0, 1.0 - abs(angle_sum - 180) / 90)
        if 145 <= angle_sum <= 215 and valid_angles and len(angles) >= 3:
            # Calculate confidence based on how close to 180 and angle distribution
            angle_diff = abs(angle_sum - 180)
            base_confidence = 0.80 - (angle_diff / 180) * 0.3
            
            # Boost if vertices are close to 3
            if len(vertices) <= 4:
                base_confidence += 0.1
                
            # Boost if solidity is good
            if features.solidity > 0.85:
                base_confidence += 0.05
                
            confidence = min(max(base_confidence, 0.5), 0.95)
            return ShapeType.TRIANGLE, confidence
            
        return None
        
    def _is_rectangle(self, features: ShapeFeatures, vertices: List[Tuple[float, float]], contour: np.ndarray) -> Optional[Tuple[ShapeType, float]]:
        """
        Check if the contour represents a rectangle or square.
        
        Args:
            features: Extracted features
            vertices: Approximated vertices
            contour: Original contour
            
        Returns:
            Optional[Tuple[ShapeType, float]]: (shape_type, confidence) or None
        """
        # Curves commonly approximate to 8 vertices; do not let their local
        # tangent angles be mistaken for rectangular corners.
        if len(vertices) != 4:
            return None
            
        # Calculate angles using rotated bounding box for better accuracy
        angles = []
        for i in range(min(len(vertices), 8)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            p3 = vertices[(i + 2) % len(vertices)]
            
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1 or norm2 < 1:
                continue
                
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))
        
        if len(angles) < 3:
            return None
            
        # Check if angles are close to 90 degrees (allow tolerance for hand-drawn)
        right_angle_count = sum(1 for angle in angles if abs(angle - 90) < self.angle_tolerance * 1.5)
        features.angle_score = self._angle_similarity(vertices, 90.0)
        
        # Require most corners to be right angles. Two right angles is also a
        # perfectly valid trapezoid and must not be promoted to a rectangle.
        if right_angle_count >= 3:
            # Check aspect ratio for square vs rectangle using rotated bounding box
            aspect_ratio = features.rotated_aspect_ratio if features.rotated_aspect_ratio > 0 else features.aspect_ratio
            
            # Ensure aspect ratio is > 0
            if aspect_ratio <= 0:
                aspect_ratio = features.aspect_ratio
                if aspect_ratio <= 0:
                    return None
            
            # Normalize aspect ratio to always be >= 1
            normalized_ar = max(aspect_ratio, 1.0 / aspect_ratio)
            
            # Check if it's a square
            if normalized_ar < 1.25:
                # Square
                confidence = 0.85 + (0.05 * (right_angle_count / 4))
                # Boost for good solidity and extent
                if features.solidity > 0.85:
                    confidence += 0.05
                if features.extent > 0.6:
                    confidence += 0.05
                return ShapeType.SQUARE, min(confidence, 0.95)
            else:
                # Rectangle
                confidence = 0.80 + (0.05 * (right_angle_count / 4))
                # Boost for good solidity and extent
                if features.solidity > 0.8:
                    confidence += 0.05
                if features.extent > 0.6:
                    confidence += 0.05
                return ShapeType.RECTANGLE, min(confidence, 0.95)
                
        return None
        
    def _is_circle(self, features: ShapeFeatures, contour: np.ndarray) -> Optional[Tuple[ShapeType, float]]:
        """
        Check if the contour represents a circle.
        
        Args:
            features: Extracted features
            contour: Original contour
            
        Returns:
            Optional[Tuple[ShapeType, float]]: (CIRCLE, confidence) or None
        """
        # Circle should have reasonable circularity (lower threshold for hand-drawn)
        if features.circularity < 0.55:
            return None
            
        # Aspect ratio should be close to 1 (allow more tolerance for hand-drawn)
        aspect_ratio = features.rotated_aspect_ratio if features.rotated_aspect_ratio > 0 else features.aspect_ratio
        if aspect_ratio <= 0:
            aspect_ratio = features.aspect_ratio
            if aspect_ratio <= 0:
                return None
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return None
            
        # Extent should be reasonable (circles fill their bounding box)
        if features.extent < 0.4:
            return None
            
        # Solidity should be good for closed shapes
        if features.solidity < 0.6:
            return None
            
        # Try to fit an ellipse and check the fit quality
        try:
            if len(contour) > 8:
                ellipse = cv2.fitEllipse(contour)
                _, (major_axis, minor_axis), _ = ellipse
                # Check if the ellipse is reasonably close to a circle
                if major_axis > 0 and minor_axis > 0:
                    ellipse_ar = max(major_axis, minor_axis) / min(major_axis, minor_axis)
                    if ellipse_ar < 1.3:
                        # It's a circle
                        confidence = 0.70 + 0.16 * features.radius_consistency
                        
                        # Boost for high circularity
                        if features.circularity > 0.8:
                            confidence += 0.10
                        elif features.circularity > 0.7:
                            confidence += 0.05
                            
                        # Boost for good aspect ratio
                        if 0.85 <= aspect_ratio <= 1.15:
                            confidence += 0.05
                            
                        # Boost for good extent and solidity
                        if features.extent > 0.65 and features.solidity > 0.85:
                            confidence += 0.05
                            
                        confidence = min(confidence, 0.98)
                        return ShapeType.CIRCLE, confidence
        except:
            pass
        
        # Fallback: use circularity and aspect ratio
        if features.circularity > 0.65 and 0.6 <= aspect_ratio <= 1.5:
            confidence = 0.66 + (features.circularity - 0.65) * 0.3 + 0.16 * features.radius_consistency
            confidence = min(confidence, 0.95)
            return ShapeType.CIRCLE, confidence
            
        return None
        
    def _is_ellipse(self, features: ShapeFeatures, contour: np.ndarray) -> Optional[Tuple[ShapeType, float]]:
        """
        Check if the contour represents an ellipse.
        
        Args:
            features: Extracted features
            contour: Original contour
            
        Returns:
            Optional[Tuple[ShapeType, float]]: (ELLIPSE, confidence) or None
        """
        # Ellipse should have decent circularity but not perfect
        if features.circularity < 0.4:
            return None
            
        if features.circularity > 0.85:
            return None
            
        # Aspect ratio should be significantly different from 1
        aspect_ratio = features.rotated_aspect_ratio if features.rotated_aspect_ratio > 0 else features.aspect_ratio
        if aspect_ratio <= 0:
            aspect_ratio = features.aspect_ratio
            if aspect_ratio <= 0:
                return None
        
        if 0.7 <= aspect_ratio <= 1.3:
            return None
            
        # Extent should be reasonable
        if features.extent < 0.3:
            return None
            
        # Solidity should be good
        if features.solidity < 0.5:
            return None
            
        # Try to fit an ellipse
        try:
            if len(contour) > 8:
                ellipse = cv2.fitEllipse(contour)
                _, (major_axis, minor_axis), _ = ellipse
                if major_axis > 0 and minor_axis > 0:
                    ellipse_ar = max(major_axis, minor_axis) / min(major_axis, minor_axis)
                    if ellipse_ar > 1.3:
                        # It's an ellipse
                        confidence = 0.75
                        
                        # Boost for better circularity
                        if features.circularity > 0.6:
                            confidence += 0.10
                            
                        # Boost for good extent
                        if features.extent > 0.5:
                            confidence += 0.05
                            
                        confidence = min(confidence, 0.90)
                        return ShapeType.ELLIPSE, confidence
        except:
            pass
        
        return None
        
    def calculate_confidence(self, shape_type: ShapeType, features: ShapeFeatures) -> float:
        """
        Calculate confidence score for a shape classification.
        
        Args:
            shape_type: Detected shape type
            features: Extracted features
            
        Returns:
            float: Confidence score between 0 and 1
        """
        base_confidence = {
            ShapeType.CIRCLE: 0.85,
            ShapeType.RECTANGLE: 0.80,
            ShapeType.SQUARE: 0.85,
            ShapeType.TRIANGLE: 0.80,
            ShapeType.ELLIPSE: 0.75,
            ShapeType.LINE: 0.80,
            ShapeType.PENTAGON: 0.84,
            ShapeType.HEXAGON: 0.84,
            ShapeType.HEPTAGON: 0.82,
            ShapeType.OCTAGON: 0.82,
            ShapeType.NONAGON: 0.80,
            ShapeType.DECAGON: 0.80,
            ShapeType.UNDECAGON: 0.78,
            ShapeType.DODECAGON: 0.78,
            ShapeType.TRAPEZOID: 0.78,
            ShapeType.PARALLELOGRAM: 0.80,
            ShapeType.RHOMBUS: 0.82,
            ShapeType.KITE: 0.76,
            ShapeType.STAR: 0.75,
            ShapeType.ARROW: 0.71,
            ShapeType.CROSS: 0.72,
            ShapeType.HEART: 0.70,
            ShapeType.SEMICIRCLE: 0.70,
            ShapeType.QUARTER_CIRCLE: 0.70,
            ShapeType.CRESCENT: 0.70,
            ShapeType.UNKNOWN: 0.20
        }.get(shape_type, 0.50)
        
        # Adjust based on geometric features
        adjustment = 0.0
        
        if shape_type == ShapeType.CIRCLE:
            adjustment = (features.circularity - 0.65) * 0.5
            if features.solidity > 0.8:
                adjustment += 0.05
        elif shape_type in [ShapeType.RECTANGLE, ShapeType.SQUARE]:
            if features.extent > 0.65:
                adjustment += 0.1
            if features.solidity > 0.8:
                adjustment += 0.05
        elif shape_type == ShapeType.TRIANGLE:
            if features.extent > 0.4:
                adjustment += 0.1
            if features.solidity > 0.8:
                adjustment += 0.05
        elif shape_type == ShapeType.LINE:
            if features.elongation > 8:
                adjustment += 0.1
        elif shape_type == ShapeType.UNKNOWN:
            adjustment = -0.1
            
        # Clamp confidence
        confidence = max(0.0, min(1.0, base_confidence + adjustment))
        
        return confidence
        
    def recognize(self, image: Union[str, Path, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        Main recognition method for 2D shapes.
        
        Args:
            image: Input image to recognize
            
        Returns:
            RecognitionResult: Recognition results with shape, confidence, and features
        """
        # Console diagnostics are optional; keeping them behind debug avoids
        # console encoding failures turning valid input into UNKNOWN.
        if self.debug:
            self._print_debug_header(image)
        
        try:
            # Preprocess image
            binary = self.preprocess_image(image)
            
            # Save binary for debug
            self._save_debug_image(binary, "04_binary")
            
            # Find contours
            contours = self.find_contours(binary)
            
            # Draw contours on a copy for debug
            if self.debug and contours:
                debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
                self._save_debug_image(debug_img, "06_contours")
            
            if not contours:
                if self.debug:
                    self._logger.debug("No valid contours found")
                result = RecognitionResult(
                    shape=ShapeType.UNKNOWN,
                    confidence=0.0,
                    features=ShapeFeatures(),
                    error="No contours found"
                )
                if self.debug:
                    self._print_result_debug(result.shape, result.confidence)
                return result
            
            # Use the largest contour
            main_contour = contours[0]
            
            # Print selected contour info
            if self.debug:
                x, y, w, h = cv2.boundingRect(main_contour)
                area = cv2.contourArea(main_contour)
                print(f"\nSelected Contour:")
                print(f"  Area: {area:.0f}")
                print(f"  Bounding Box: ({x}, {y}) - {w}x{h}")
                touches = self._contour_touches_border(main_contour, binary.shape)
                print(f"  Touches border: {touches}")
            
            # Draw selected contour for debug
            if self.debug:
                debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, [main_contour], -1, (0, 255, 255), 3)
                self._save_debug_image(debug_img, "07_selected_contour")
            
            # Recognize shape
            shape_type, confidence, features, vertices = self.recognize_shape(main_contour)
            
            # Retain the candidate-specific evidence rather than overwriting it
            # with a generic type prior.
            final_confidence = max(confidence, self.calculate_confidence(shape_type, features))
            
            # Print features and result
            if self.debug:
                self._print_features_debug(features)
                self._print_result_debug(shape_type, final_confidence)
            
            if self.debug:
                self._logger.debug(f"Recognition result: {shape_type.value}, confidence: {final_confidence:.3f}")
                self._logger.debug(f"  Area: {features.area:.0f}, Vertices: {features.vertices}")
                self._logger.debug(f"  Aspect Ratio: {features.aspect_ratio:.3f}, Circularity: {features.circularity:.3f}")
                self._logger.debug(f"  Solidity: {features.solidity:.3f}, Extent: {features.extent:.3f}")
            
            return RecognitionResult(
                shape=shape_type,
                confidence=final_confidence,
                features=features,
                raw_vertices=vertices,
                contour_points=[(float(point[0][0]), float(point[0][1])) for point in main_contour],
            )
            
        except Exception as e:
            self._logger.error(f"Recognition failed: {str(e)}")
            print(f"\n❌ ERROR: Recognition failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return RecognitionResult(
                shape=ShapeType.UNKNOWN,
                confidence=0.0,
                features=ShapeFeatures(),
                error=str(e)
            )
            
    def recognize_batch(self, images: List[Union[str, Path, np.ndarray, Image.Image]]) -> List[RecognitionResult]:
        """
        Recognize multiple images in batch.
        
        Args:
            images: List of input images
            
        Returns:
            List[RecognitionResult]: List of recognition results
        """
        return [self.recognize(img) for img in images]
        
    def get_supported_shapes(self) -> List[str]:
        """
        Get list of supported shape types.
        
        Returns:
            List[str]: List of supported shape names
        """
        return [shape.value for shape in ShapeType]
        
    def __repr__(self) -> str:
        """
        String representation of the ShapeRecognizer.
        
        Returns:
            str: Human-readable representation
        """
        return f"ShapeRecognizer(min_area={self.min_contour_area}, shapes={len(self.get_supported_shapes())})"


def generate_test_shape(shape_type: str, size: int = 400, thickness: int = 10) -> np.ndarray:
    """
    Generate a test image for a given shape.
    
    Args:
        shape_type: Type of shape to generate
        size: Image size in pixels
        thickness: Stroke thickness in pixels
        
    Returns:
        np.ndarray: Generated image
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    half = size // 3
    
    if shape_type == "circle":
        cv2.circle(img, center, half, (255, 255, 255), thickness)
    elif shape_type == "square":
        cv2.rectangle(img, (center[0] - half, center[1] - half), 
                      (center[0] + half, center[1] + half), (255, 255, 255), thickness)
    elif shape_type == "rectangle":
        cv2.rectangle(img, (center[0] - half, center[1] - half//2), 
                      (center[0] + half, center[1] + half//2), (255, 255, 255), thickness)
    elif shape_type == "triangle":
        pts = np.array([
            [center[0], center[1] - half],
            [center[0] + half, center[1] + half],
            [center[0] - half, center[1] + half]
        ], np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 255), thickness)
    elif shape_type == "ellipse":
        cv2.ellipse(img, center, (half, half//2), 0, 0, 360, (255, 255, 255), thickness)
    elif shape_type == "line":
        cv2.line(img, (50, center[1]), (size - 50, center[1]), (255, 255, 255), thickness)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    return img


def generate_imperfect_test_shape(shape_type: str, size: int = 400) -> np.ndarray:
    """
    Generate an imperfect test image for a given shape to simulate hand-drawn.
    
    Args:
        shape_type: Type of shape to generate
        size: Image size in pixels
        
    Returns:
        np.ndarray: Generated image
    """
    img = generate_test_shape(shape_type, size, thickness=8)
    
    # Add some noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Slight blur to simulate anti-aliasing
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Random small gaps (simulate shaky hand)
    if np.random.random() > 0.5:
        gap_size = np.random.randint(5, 15)
        gap_pos = np.random.randint(0, size)
        img[gap_pos:gap_pos + gap_size, :] = 0
    
    return img


def test_shape_recognizer():
    """
    Comprehensive test function for the ShapeRecognizer.
    """
    import os
    
    # Create test directories
    test_dir = Path("test_shapes")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample images for testing
    recognizer = ShapeRecognizer(debug=True)
    
    print("=" * 70)
    print("Shape Recognizer Test")
    print("=" * 70)
    print(f"Supported shapes: {recognizer.get_supported_shapes()}")
    print()
    
    # Generate and test clean shapes
    print("Generating and testing clean shapes...")
    print("-" * 50)
    
    clean_shapes = ["circle", "square", "rectangle", "triangle", "ellipse", "line"]
    
    for shape in clean_shapes:
        # Generate clean image
        img = generate_test_shape(shape, size=400, thickness=8)
        img_path = test_dir / f"clean_{shape}.png"
        Image.fromarray(img).save(img_path)
        
        # Recognize
        result = recognizer.recognize(img_path)
        
        print(f"\n📷 Clean {shape}:")
        print(f"  Expected: {shape}")
        print(f"  Detected: {result.shape.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  PASS/FAIL: {'✅ PASS' if result.shape.value == shape else '❌ FAIL'}")
        print(f"  Vertices: {result.features.vertices}")
        print(f"  Aspect Ratio: {result.features.aspect_ratio:.2f}")
        print(f"  Circularity: {result.features.circularity:.2f}")
        print(f"  Solidity: {result.features.solidity:.2f}")
    
    # Test with imperfect images
    print("\n" + "-" * 50)
    print("Testing imperfect hand-drawn-like shapes...")
    print("-" * 50)
    
    for shape in clean_shapes:
        img = generate_imperfect_test_shape(shape, size=400)
        img_path = test_dir / f"imperfect_{shape}.png"
        Image.fromarray(img).save(img_path)
        
        result = recognizer.recognize(img_path)
        
        print(f"\n📷 Imperfect {shape}:")
        print(f"  Expected: {shape}")
        print(f"  Detected: {result.shape.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  PASS/FAIL: {'✅ PASS' if result.shape.value == shape else '❌ FAIL'}")
        print(f"  Vertices: {result.features.vertices}")
        print(f"  Aspect Ratio: {result.features.aspect_ratio:.2f}")
        print(f"  Circularity: {result.features.circularity:.2f}")
        print(f"  Solidity: {result.features.solidity:.2f}")
    
    # Test with existing images in test_shapes
    print("\n" + "-" * 50)
    print("Testing existing images in test_shapes directory...")
    print("-" * 50)
    
    test_images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    # Filter out generated images to avoid duplicates
    test_images = [img for img in test_images if "clean_" not in img.name and "imperfect_" not in img.name]
    
    if test_images:
        for img_path in test_images:
            try:
                result = recognizer.recognize(img_path)
                # Try to determine expected shape from filename
                expected = "unknown"
                for shape in clean_shapes:
                    if shape in img_path.name.lower():
                        expected = shape
                        break
                
                print(f"\n📷 Image: {img_path.name}")
                print(f"  Expected: {expected}")
                print(f"  Detected: {result.shape.value}")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  PASS/FAIL: {'✅ PASS' if result.shape.value == expected else '⚠️  CHECK'}")
                print(f"  Features:")
                print(f"    Area: {result.features.area:.0f}")
                print(f"    Perimeter: {result.features.perimeter:.1f}")
                print(f"    Vertices: {result.features.vertices}")
                print(f"    Width x Height: {result.features.width:.0f} x {result.features.height:.0f}")
                print(f"    Aspect Ratio: {result.features.aspect_ratio:.2f}")
                print(f"    Circularity: {result.features.circularity:.2f}")
                print(f"    Solidity: {result.features.solidity:.2f}")
                print(f"    Extent: {result.features.extent:.2f}")
                if result.error:
                    print(f"  Error: {result.error}")
            except Exception as e:
                print(f"❌ Failed to test {img_path.name}: {e}")
    else:
        print("\nNo additional test images found.")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_shape_recognizer()

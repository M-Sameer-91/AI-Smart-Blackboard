"""
preprocessor.py - Image Preprocessing Module for AI Smart Blackboard

This module provides advanced image preprocessing for OCR optimization.
It applies a comprehensive pipeline of image enhancement techniques to
significantly improve OCR accuracy for handwritten and printed text.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Union
import logging


class ImagePreprocessor:
    """
    Advanced image preprocessor for OCR optimization.
    
    This class applies a comprehensive preprocessing pipeline to prepare
    images for optimal OCR performance. It handles various image quality
    issues including low contrast, noise, uneven lighting, and small text.
    
    The preprocessing pipeline includes:
    1. Grayscale conversion
    2. Contrast enhancement
    3. Gaussian blur for noise reduction
    4. Adaptive thresholding for binarization
    5. Morphological operations for cleaning
    6. Noise removal
    7. Resizing to OCR-friendly dimensions
    8. Content centering
    """
    
    def __init__(
        self,
        contrast_factor: float = 2.0,
        blur_ksize: Tuple[int, int] = (3, 3),
        adaptive_block_size: int = 11,
        adaptive_constant: int = 2,
        morph_kernel_size: Tuple[int, int] = (3, 3),
        target_width: int = 1200,
        target_height: int = 800,
        min_text_height: int = 50,
        min_whitespace: int = 50
    ) -> None:
        """
        Initialize the ImagePreprocessor with configurable parameters.
        
        Args:
            contrast_factor: Multiplier for contrast enhancement (default: 2.0)
            blur_ksize: Kernel size for Gaussian blur (default: (3,3))
            adaptive_block_size: Block size for adaptive thresholding (default: 11)
            adaptive_constant: Constant subtracted from mean (default: 2)
            morph_kernel_size: Kernel size for morphological operations (default: (3,3))
            target_width: Target width for resizing (default: 1200)
            target_height: Target height for resizing (default: 800)
            min_text_height: Minimum text height in pixels (default: 50)
            min_whitespace: Minimum whitespace padding in pixels (default: 50)
        """
        self.contrast_factor = contrast_factor
        self.blur_ksize = blur_ksize
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_constant = adaptive_constant
        self.morph_kernel_size = morph_kernel_size
        self.target_width = target_width
        self.target_height = target_height
        self.min_text_height = min_text_height
        self.min_whitespace = min_whitespace
        
        # Initialize morphological kernel
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.morph_kernel_size
        )
        
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file and convert to OpenCV format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Image as numpy array (BGR format)
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image using OpenCV
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def convert_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Grayscale image
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        # Apply CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_factor,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(image)
        
        # Additional gamma correction for better visibility
        gamma = 1.2
        look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, look_up_table)
        
        return enhanced
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise using Gaussian blur and median filtering.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Denoised image
        """
        # Apply Gaussian blur for general noise reduction
        blurred = cv2.GaussianBlur(image, self.blur_ksize, 0)
        
        # Apply median filter for salt-and-pepper noise reduction
        denoised = cv2.medianBlur(blurred, 3)
        
        # Apply bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for binarization.
        
        Args:
            image: Grayscale image
            
        Returns:
            np.ndarray: Binary image
        """
        # Ensure block size is odd
        block_size = self.adaptive_block_size if self.adaptive_block_size % 2 == 1 else self.adaptive_block_size + 1
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_constant
        )
        
        return binary
    
    def morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean the binary image.
        
        Args:
            image: Binary image
            
        Returns:
            np.ndarray: Morphologically cleaned image
        """
        # Apply morphological opening (erosion followed by dilation)
        # Removes small noise particles
        opened = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            self._morph_kernel,
            iterations=1
        )
        
        # Apply morphological closing (dilation followed by erosion)
        # Fills small gaps in text
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=1
        )
        
        return closed
    
    def remove_small_noise(self, image: np.ndarray, min_area: int = 20) -> np.ndarray:
        """
        Remove small noise components from the binary image.
        
        Args:
            image: Binary image
            min_area: Minimum area threshold for component retention
            
        Returns:
            np.ndarray: Cleaned binary image
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            image,
            connectivity=8
        )
        
        # Create output image
        result = np.zeros_like(image)
        
        # Keep only components larger than min_area
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 255
        
        return result
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to OCR-friendly dimensions while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate scale to fit target dimensions
        scale_width = self.target_width / width
        scale_height = self.target_height / height
        scale = min(scale_width, scale_height, 2.0)  # Cap scaling at 2x
        
        # Ensure minimum size for OCR
        if width < self.target_width * 0.5 or height < self.target_height * 0.5:
            scale = max(scale, 1.5)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Ensure dimensions are at least minimum
        new_width = max(new_width, 400)
        new_height = max(new_height, 200)
        
        # Resize image
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        return resized
    
    def center_content(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and center handwritten content within the image.
        
        Args:
            image: Binary image
            
        Returns:
            np.ndarray: Image with centered content
        """
        # Find contours of content
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # Find bounding box of all content
        x_coords = []
        y_coords = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
        
        if not x_coords or not y_coords:
            return image
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Add padding
        padding = self.min_whitespace
        min_x = max(0, min_x - padding)
        max_x = min(image.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image.shape[0], max_y + padding)
        
        # Crop to content
        cropped = image[min_y:max_y, min_x:max_x]
        
        # Calculate padding for centering
        height, width = cropped.shape
        pad_top = (self.target_height - height) // 2
        pad_bottom = self.target_height - height - pad_top
        pad_left = (self.target_width - width) // 2
        pad_right = self.target_width - width - pad_left
        
        # Ensure padding is not negative
        pad_top = max(0, pad_top)
        pad_bottom = max(0, pad_bottom)
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)
        
        # Apply padding
        centered = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=255
        )
        
        return centered
    
    def enhance_text_clarity(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text clarity using sharpening and edge enhancement.
        
        Args:
            image: Binary image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Create sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        
        # Apply sharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Apply threshold to maintain binary nature
        _, sharpened = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
        
        # ximgproc is only supplied by opencv-contrib.  Do not make OCR fail
        # on the normal opencv-python package just to thin an image.
        return sharpened

    def preprocess_blackboard_text(self, image: np.ndarray) -> Image.Image:
        """Create a clean, cropped black-text-on-white OCR image from canvas ink."""
        gray = self.convert_grayscale(image)
        _, bright_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_ink = cv2.bitwise_not(bright_ink)

        # Select the sparse polarity: board drawings are normally sparse,
        # regardless of whether the board is black or white.
        bright_ratio = np.count_nonzero(bright_ink) / bright_ink.size
        dark_ratio = np.count_nonzero(dark_ink) / dark_ink.size
        candidates = [(bright_ink, bright_ratio), (dark_ink, dark_ratio)]
        plausible = [item for item in candidates if 0.0005 <= item[1] <= 0.45]
        ink = min(plausible or candidates, key=lambda item: item[1])[0]

        points = cv2.findNonZero(ink)
        if points is None:
            raise ValueError("Empty drawing: no ink pixels were found")
        x, y, w, h = cv2.boundingRect(points)
        padding = max(20, int(max(w, h) * 0.08))
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(ink.shape[1], x + w + padding), min(ink.shape[0], y + h + padding)
        cropped = ink[y0:y1, x0:x1]

        # Tesseract expects dark glyphs on a light page. Upscale without
        # erosion so the three-pixel board pen remains legible.
        scale = max(3.0, 900.0 / max(cropped.shape[1], 1))
        resized = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        page = cv2.copyMakeBorder(resized, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
        return Image.fromarray(cv2.bitwise_not(page))
    
    def preprocess(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Execute the complete preprocessing pipeline on an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Image.Image: Preprocessed PIL Image ready for OCR
            
        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If preprocessing fails
        """
        try:
            # Load image
            logging.info(f"Loading image: {image_path}")
            image = self.load_image(image_path)
            
            logging.info("Preparing blackboard ink for OCR")
            pil_image = self.preprocess_blackboard_text(image)
            
            logging.info("Preprocessing completed successfully")
            return pil_image
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing failed: {str(e)}") from e
    
    def preprocess_and_save(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Preprocess an image and save the result.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the preprocessed image (optional)
            
        Returns:
            Path: Path to the saved preprocessed image
        """
        # Perform preprocessing
        processed = self.preprocess(image_path)
        
        # Generate output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_processed.png"
        
        # Save the processed image
        processed.save(output_path)
        
        return Path(output_path)
    
    def get_processing_summary(self, image_path: Union[str, Path]) -> dict:
        """
        Get a summary of the preprocessing applied to an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            dict: Dictionary containing preprocessing details
        """
        image = self.load_image(image_path)
        original_shape = image.shape
        
        # Perform preprocessing
        processed = self.preprocess(image_path)
        processed_shape = processed.size
        
        return {
            "original_size": original_shape[:2],
            "processed_size": processed_shape,
            "contrast_factor": self.contrast_factor,
            "blur_kernel": self.blur_ksize,
            "adaptive_block": self.adaptive_block_size,
            "morph_kernel": self.morph_kernel_size,
            "target_dimensions": (self.target_width, self.target_height),
            "status": "success"
        }

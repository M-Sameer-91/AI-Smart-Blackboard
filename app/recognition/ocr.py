"""
ocr.py - Optical Character Recognition Module with Intelligent Input Detection

This module provides advanced OCR capabilities with automatic input type detection
and optimized preprocessing for different content types including single characters,
mathematical symbols, words, and sentences.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np

from app.recognition.preprocessor import ImagePreprocessor


class OCRProcessor:
    """
    Advanced OCR processor with intelligent input type detection.
    
    This class handles the complete OCR pipeline including:
    - Automatic detection of input type (character, word, math, sentence)
    - Optimized preprocessing for different content types
    - Dynamic Tesseract configuration selection
    - Confidence scoring
    - Comprehensive error handling
    
    The model is loaded once and reused for all recognition requests.
    """
    
    def __init__(self, model_name: str = "default", save_processed_images: bool = True) -> None:
        """
        Initialize the OCR processor with Tesseract and preprocessor.
        
        Args:
            model_name: Model identifier (kept for compatibility)
        """
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None
        self._preprocessor: Optional[ImagePreprocessor] = None
        self._processed_dir = Path("app/data/processed")
        self._save_processed_images = save_processed_images
        
        # Mathematical symbols and patterns
        self._math_symbols = {
            'sqrt': '√',
            'pi': 'π',
            'sigma': 'Σ',
            'integral': '∫',
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'theta': 'θ',
            'plus_minus': '±',
            'multiply': '×',
            'divide': '÷',
            'equals': '=',
            'approx': '≈',
            'less_than': '<',
            'greater_than': '>',
            'less_equal': '≤',
            'greater_equal': '≥',
            'not_equal': '≠'
        }
        
        # Mathematical patterns for detection
        self._math_patterns = [
            r'[+\\-*/=×÷±]',  # Operators
            r'\^[0-9²³⁴⁵⁶⁷⁸⁹]',  # Exponents
            r'[xXyYzZ][²³⁴⁵⁶⁷⁸⁹]?',  # Variables with exponents
            r'[sin|cos|tan|log|ln|exp]\(',  # Functions
            r'[√πΣ∫αβγθ]',  # Greek and math symbols
            r'\d+[/⁄]\\d+',  # Fractions
            r'[a-zA-Z]\([^)]+\)',  # Function notation
            r'[0-9]+[a-zA-Z]+',  # Numbers with variables (5x)
        ]
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        # Initialize preprocessor
        self._initialize_preprocessor()
        
        # Load Tesseract
        self._load_tesseract()
        
        # Create processed directory
        self._create_processed_directory()
        
    def _initialize_preprocessor(self) -> None:
        """
        Initialize the ImagePreprocessor with optimized parameters.
        """
        try:
            self._preprocessor = ImagePreprocessor(
                contrast_factor=2.5,
                blur_ksize=(3, 3),
                adaptive_block_size=15,
                adaptive_constant=3,
                morph_kernel_size=(3, 3),
                target_width=1600,
                target_height=1200,
                min_text_height=40,
                min_whitespace=50
            )
            self._logger.info("ImagePreprocessor initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize preprocessor: {str(e)}")
            self._preprocessor = None
            
    def _load_tesseract(self) -> None:
        """
        Load and initialize the Tesseract OCR engine.
        """
        self._logger.info("Loading Tesseract OCR engine...")
        
        # Set Tesseract path
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            version = pytesseract.get_tesseract_version()
            self._is_loaded = True
            self._load_error = None
            self._logger.info(f"✅ Tesseract loaded successfully")
            self._logger.info(f"   Version: {version}")
        except Exception as e:
            self._load_error = f"Tesseract not found at: {tesseract_path}\nPlease install Tesseract OCR"
            self._logger.error(self._load_error)
            print(self._load_error, file=sys.stderr)
            
    def _create_processed_directory(self) -> None:
        """
        Create the directory for storing processed images.
        """
        try:
            self._processed_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Processed directory ready: {self._processed_dir}")
        except Exception as e:
            self._logger.error(f"Failed to create processed directory: {str(e)}")
            
    def _generate_processed_filename(self) -> str:
        """
        Generate a timestamp-based filename for processed images.
        
        Returns:
            str: Formatted filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"processed_{timestamp}.png"
        
    def _detect_input_type(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect the type of input content in the image.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Dict[str, Any]: Detection results with type and confidence
        """
        height, width = image.shape[:2]
        
        # The preprocessor deliberately produces *black ink on a white page*.
        # Using THRESH_BINARY here made the white page itself the only contour,
        # so every drawing was classified as ``general_text``.  Detect the ink
        # instead; this is also robust to RGB/RGBA input supplied by callers.
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"type": "empty", "confidence": 1.0}
        
        # Calculate content area
        total_area = 0
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 3 and h > 3:  # Filter noise
                area = w * h
                total_area += area
                bounding_boxes.append((x, y, w, h))
        
        if total_area == 0:
            return {"type": "empty", "confidence": 1.0}
        
        # Calculate overall bounding box
        if bounding_boxes:
            min_x = min(box[0] for box in bounding_boxes)
            max_x = max(box[0] + box[2] for box in bounding_boxes)
            min_y = min(box[1] for box in bounding_boxes)
            max_y = max(box[1] + box[3] for box in bounding_boxes)
            
            content_width = max_x - min_x
            content_height = max_y - min_y
            aspect_ratio = content_width / content_height if content_height > 0 else 0
        else:
            content_width = width
            content_height = height
            aspect_ratio = width / height
        
        # Detect content type based on characteristics
        self._logger.debug(
            "OCR ink analysis: components=%d, bbox=%dx%d, page=%dx%d",
            len(bounding_boxes), content_width, content_height, width, height,
        )

        # A character may consist of several contours (for example, i or j),
        # so use the combined ink bounding box rather than len(contours) == 1.
        if len(bounding_boxes) <= 3 and content_width <= content_height * 1.25:
            return {"type": "single_character", "confidence": 0.95}
        
        # Single word detection (few connected components, narrow)
        if len(bounding_boxes) < 12 and content_width > content_height and content_width / content_height > 1.5:
            return {"type": "single_word", "confidence": 0.85}
        
        # Mathematical expression detection
        math_indicators = 0
        binary_text = cv2.bitwise_not(binary)  # Invert for visualization
        for pattern in self._math_patterns:
            # Simple heuristic: look for mathematical patterns in the image
            # In practice, we'll use Tesseract to detect these
            pass
        
        # Multiple lines detection
        if len(bounding_boxes) >= 12 and content_width / content_height > 2:
            return {"type": "multi_line", "confidence": 0.8}
        
        # Default to general text
        return {"type": "general_text", "confidence": 0.75}
        
    def _optimize_preprocessing(self, input_type: str, image: Image.Image) -> Image.Image:
        """
        Apply input-type-specific preprocessing optimizations.
        
        Args:
            input_type: Detected input type
            image: PIL Image to preprocess
            
        Returns:
            Image.Image: Optimized PIL Image
        """
        # Convert to numpy for OpenCV operations
        img_np = np.array(image)
        
        if input_type == "single_character":
            # Aggressive preprocessing for single characters
            # Increase contrast more
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(3.0)
            
            # Sharpen more
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.SHARPEN)
            
            # Ensure high resolution
            width, height = image.size
            if width < 200 or height < 200:
                scale = max(200 / width, 200 / height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale and threshold
            img_np = np.array(image.convert('L'))
            _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            image = Image.fromarray(binary)
            
        elif input_type == "single_word":
            # Moderate preprocessing for words
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.5)
            image = image.filter(ImageFilter.SHARPEN)
            
            img_np = np.array(image.convert('L'))
            _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image = Image.fromarray(binary)
            
        elif input_type == "multi_line" or input_type == "general_text":
            # Standard preprocessing for text blocks
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)
            
        return image
        
    def _get_ocr_config(self, input_type: str) -> Tuple[str, str]:
        """
        Get optimized Tesseract configuration for the input type.
        
        Args:
            input_type: Detected input type
            
        Returns:
            Tuple[str, str]: (psm_mode, additional_config)
        """
        configs = {
            "single_character": {
                "psm": "--psm 10",  # Single character
                "extra": "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789²³√πΣ∫±×÷αβγθ"
            },
            "single_word": {
                "psm": "--psm 8",  # Single word
                "extra": ""
            },
            "multi_line": {
                "psm": "--psm 6",  # Block of text
                "extra": ""
            },
            "general_text": {
                "psm": "--psm 6",  # Block of text
                "extra": ""
            },
            "empty": {
                "psm": "--psm 6",
                "extra": ""
            }
        }
        
        config = configs.get(input_type, configs["general_text"])
        return config["psm"], config["extra"]
        
    def _calculate_confidence(self, text: str, input_type: str) -> float:
        """
        Calculate confidence score based on input type and recognized text.
        
        Args:
            text: Recognized text
            input_type: Detected input type
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not text:
            return 0.0
            
        confidence = 0.5  # Base confidence
        
        # Single character confidence
        if input_type == "single_character":
            if len(text) == 1:
                if text.isalpha() or text.isdigit():
                    confidence = 0.95
                elif any(unicode_char in text for unicode_char in self._math_symbols.values()):
                    confidence = 0.90
                else:
                    confidence = 0.70
            elif len(text) <= 2 and text.isdigit():
                confidence = 0.60
            else:
                confidence = 0.30
                
        # Single word confidence
        elif input_type == "single_word":
            word_count = len(text.split())
            if word_count == 1:
                confidence = 0.85
                if text.isalpha():
                    confidence = 0.90
                elif re.search(r'[a-zA-Z][²³]?', text):
                    confidence = 0.85
            elif word_count <= 3:
                confidence = 0.75
            else:
                confidence = 0.60
                
        # General text confidence
        elif input_type == "general_text" or input_type == "multi_line":
            word_count = len(text.split())
            if word_count >= 3:
                confidence = 0.80
            elif word_count >= 1:
                confidence = 0.70
            else:
                confidence = 0.40
                
        # Length-based adjustment
        text_length = len(text.strip())
        if text_length > 0 and text_length < 3:
            confidence = min(confidence + 0.10, 0.95)
            
        # Mathematical content boost
        if any(pattern in text for pattern in ['=', '+', '-', '*', '/', '×', '÷', '²', '√', 'π', 'Σ', '∫']):
            confidence = min(confidence + 0.10, 1.0)
            
        return min(confidence, 1.0)
        
    def _preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Preprocess the image using ImagePreprocessor.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Optional[Image.Image]: Preprocessed PIL Image, or None if failed
        """
        try:
            self._logger.info("Preprocessing image...")
            
            if not self._preprocessor:
                self._logger.warning("Preprocessor not available, using fallback preprocessing")
                return self._fallback_preprocess(image_path)
            
            # Preprocess the image
            processed_image = self._preprocessor.preprocess(image_path)
            
            # Font evaluation and other controlled callers may explicitly opt
            # out so their artifacts never mix with runtime OCR output.
            if self._save_processed_images:
                processed_filename = self._generate_processed_filename()
                processed_path = self._processed_dir / processed_filename
                processed_image.save(processed_path)
                self._logger.info(f"Processed image saved: {processed_path}")
            
            return processed_image
            
        except Exception as e:
            self._logger.error(f"Preprocessing failed: {str(e)}")
            return self._fallback_preprocess(image_path)
            
    def _fallback_preprocess(self, image_path: Path) -> Optional[Image.Image]:
        """
        Fallback preprocessing method if ImagePreprocessor fails.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Optional[Image.Image]: Preprocessed PIL Image, or None if failed
        """
        try:
            self._logger.info("Using fallback preprocessing...")
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert('L')
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)
            
            processed_filename = f"fallback_{self._generate_processed_filename()}"
            processed_path = self._processed_dir / processed_filename
            image.save(processed_path)
            self._logger.info(f"Fallback processed image saved: {processed_path}")
            
            return image
            
        except Exception as e:
            self._logger.error(f"Fallback preprocessing failed: {str(e)}")
            return None

    def _drawing_has_ink(self, image_path: Path) -> bool:
        """Return whether an input canvas contains a meaningful ink region."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        # One of these polarities is sparse for a real drawing; a blank black or
        # blank white canvas has no sparse foreground at all.
        bright = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
        dark = cv2.threshold(image, 223, 255, cv2.THRESH_BINARY_INV)[1]
        candidates = [np.count_nonzero(bright), np.count_nonzero(dark)]
        ink_pixels = min(candidates)
        self._logger.debug("OCR canvas ink pixels: %d", ink_pixels)
        return ink_pixels >= 20
            
    def _clean_ocr_output(self, text: str) -> str:
        """
        Clean and format the OCR output text.
        
        Args:
            text: Raw OCR output
            
        Returns:
            str: Cleaned and formatted text
        """
        if not text:
            return ""
        
        # Remove extra spaces
        cleaned = ' '.join(text.split())
        
        # Remove empty lines
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Remove duplicate spaces
        cleaned = ' '.join(cleaned.split())

        # A handwriting font or a slightly clipped first stroke is frequently
        # read as a stray opening quote.  Do not remove punctuation within the
        # text (which can be meaningful for maths), only isolated edge marks.
        cleaned = re.sub(r"^[`'‘’\"“”]+|[`'‘’\"“”]+$", "", cleaned).strip()
        
        return cleaned.strip()
        
    def extract_text(self, image_path: Path) -> str:
        """
        Extract text from image with intelligent type detection.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            str: Recognized text from the image
        """
        result = self.extract_text_with_confidence(image_path)
        return result.get("text", "")
        
    def extract_text_with_confidence(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract text from image with confidence score and metadata.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            Dict[str, Any]: Dictionary containing text, confidence, and metadata
        """
        try:
            self._logger.info("Starting OCR extraction with intelligent detection...")
            
            # Validate image path
            if not image_path.exists():
                error_msg = f"Image file not found: {image_path}"
                self._logger.error(error_msg)
                return {
                    "text": f"[OCR Error] {error_msg}",
                    "confidence": 0.0,
                    "mode": "error"
                }
            
            # Check if Tesseract is loaded
            if not self._is_loaded:
                error_msg = self._load_error or "Tesseract not loaded"
                self._logger.error(error_msg)
                return {
                    "text": f"[OCR Error] {error_msg}",
                    "confidence": 0.0,
                    "mode": "error"
                }

            if not self._drawing_has_ink(image_path):
                return {
                    "text": "Empty drawing: draw text before recognition.",
                    "confidence": 0.0,
                    "mode": "empty",
                }
            
            # Preprocess the image
            processed_image = self._preprocess_image(image_path)
            if processed_image is None:
                error_msg = "Failed to preprocess image"
                self._logger.error(error_msg)
                return {
                    "text": f"[OCR Error] {error_msg}",
                    "confidence": 0.0,
                    "mode": "error"
                }
            
            # Detect input type
            img_np = np.array(processed_image.convert('L'))
            detection_result = self._detect_input_type(img_np)
            input_type = detection_result["type"]
            self._logger.info(f"Detected input type: {input_type}")
            
            # Optimize preprocessing for input type
            optimized_image = self._optimize_preprocessing(input_type, processed_image)
            
            # Get OCR configuration
            psm_mode, extra_config = self._get_ocr_config(input_type)
            self._logger.info(f"Using OCR config: {psm_mode} {extra_config}")
            
            # Run OCR
            self._logger.info("Running OCR...")
            
            # Try primary configuration.  image_to_data gives Tesseract's real
            # word confidence, unlike the former text-length heuristic.
            try:
                config = f"{psm_mode} {extra_config}"
                text = pytesseract.image_to_string(
                    optimized_image,
                    lang='eng',
                    config=config
                )
            except Exception as e:
                self._logger.warning(f"Primary OCR attempt failed: {str(e)}")
                text = pytesseract.image_to_string(optimized_image, lang='eng')
            
            # Clean the output
            cleaned_text = self._clean_ocr_output(text)
            
            # If no text and it's single character, try more aggressive settings
            if not cleaned_text and input_type == "single_character":
                self._logger.info("Trying fallback OCR for single character...")
                try:
                    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789²³√πΣ∫±×÷αβγθ"
                    text = pytesseract.image_to_string(
                        optimized_image,
                        lang='eng',
                        config=config
                    )
                    cleaned_text = self._clean_ocr_output(text)
                except Exception as e:
                    self._logger.warning(f"Fallback OCR attempt failed: {str(e)}")
            
            # If still no text, try even more aggressive approach
            if not cleaned_text:
                self._logger.info("Trying aggressive OCR settings...")
                try:
                    # Invert image for better contrast
                    inverted = Image.eval(optimized_image, lambda x: 255 - x)
                    config = "--psm 10"
                    text = pytesseract.image_to_string(
                        inverted,
                        lang='eng',
                        config=config
                    )
                    cleaned_text = self._clean_ocr_output(text)
                except Exception as e:
                    self._logger.warning(f"Aggressive OCR attempt failed: {str(e)}")
            
            # Prefer Tesseract's measured confidence when it has word data.
            # Some PSM modes do not return usable data, in which case retain the
            # existing conservative heuristic as a fallback.
            confidence = self._calculate_confidence(cleaned_text, input_type)
            try:
                data = pytesseract.image_to_data(
                    optimized_image, lang="eng", config=f"{psm_mode} {extra_config}",
                    output_type=pytesseract.Output.DICT,
                )
                values = [float(value) for value in data.get("conf", []) if float(value) >= 0]
                if values:
                    confidence = max(0.0, min(1.0, sum(values) / len(values) / 100.0))
            except Exception as exc:
                self._logger.debug("Could not obtain OCR confidence: %s", exc)
            
            if not cleaned_text:
                self._logger.warning("No text recognized")
                return {
                    "text": "No text recognized",
                    "confidence": 0.0,
                    "mode": input_type
                }
            
            self._logger.info(f"OCR completed. Text: '{cleaned_text}', Confidence: {confidence:.2f}")
            
            return {
                "text": cleaned_text,
                "confidence": confidence,
                "mode": input_type,
                "detection_confidence": detection_result["confidence"],
                "character_count": len(cleaned_text),
                "word_count": len(cleaned_text.split())
            }
            
        except Exception as e:
            error_msg = f"OCR failed: {str(e)}"
            self._logger.error(error_msg)
            return {
                "text": f"[OCR Error] {error_msg}",
                "confidence": 0.0,
                "mode": "error"
            }
            
    def is_model_loaded(self) -> bool:
        """
        Check if the OCR model is loaded and ready.
        
        Returns:
            bool: True if loaded and ready, False otherwise
        """
        return self._is_loaded
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OCR system.
        
        Returns:
            Dict[str, Any]: Dictionary containing system information
        """
        return {
            "model": "Tesseract OCR with Intelligent Detection",
            "is_loaded": self._is_loaded,
            "error": self._load_error if not self._is_loaded else None,
            "preprocessor": "ImagePreprocessor" if self._preprocessor else "Fallback",
            "processed_dir": str(self._processed_dir),
            "supported_modes": [
                "single_character",
                "single_word",
                "multi_line",
                "general_text"
            ]
        }
        
    def get_processed_directory(self) -> Path:
        """
        Get the directory path where processed images are saved.
        
        Returns:
            Path: Path to the processed images directory
        """
        return self._processed_dir
        
    def __repr__(self) -> str:
        """
        String representation of the OCRProcessor.
        
        Returns:
            str: Human-readable representation
        """
        status = "loaded" if self._is_loaded else "not loaded"
        preprocessor_status = "with" if self._preprocessor else "without"
        return f"OCRProcessor(model='Tesseract-Intelligent', preprocessor='{preprocessor_status}', status='{status}')"

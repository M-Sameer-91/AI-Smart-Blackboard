"""
Handwriting Recognition Preprocessing Module

This module provides deterministic preprocessing for handwritten character images.
It handles various input formats and produces consistent output for model training.
"""

import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
import torch


class HandwritingPreprocessor:
    """
    Preprocess handwritten character images for recognition.
    
    This preprocessor handles:
    - Grayscale/RGB/Canvas inputs
    - Light and dark backgrounds
    - Centered and off-center characters
    - Aspect ratio preservation
    - Configurable target size
    
    Args:
        target_size: Final square image size (default: 64)
        border_ratio: Fraction of border to add around character (default: 0.1)
        background_value: Grayscale value for background (0=black, 255=white)
    """
    
    def __init__(
        self,
        target_size: int = 64,
        border_ratio: float = 0.1,
        background_value: int = 255
    ):
        self.target_size = target_size
        self.border_ratio = max(0.0, min(0.3, border_ratio))
        self.background_value = background_value
    
    def preprocess(
        self,
        image: Union[str, Image.Image, np.ndarray],
        return_tensor: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess a handwritten character image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_tensor: If True, return PyTorch tensor; else return numpy array
        
        Returns:
            Preprocessed image as tensor (1, 1, H, W) or array (H, W)
        
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        # Load image if path provided
        if isinstance(image, str):
            try:
                img = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image from {image}: {e}")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array for processing
        arr = np.array(img, dtype=np.uint8)
        
        # Detect and handle polarity
        arr = self._handle_polarity(arr)
        
        # Find character region
        bbox = self._find_character_region(arr)
        
        # Extract and crop character
        cropped = self._crop_character(arr, bbox)
        
        # Add border
        bordered = self._add_border(cropped)
        
        # Resize preserving aspect ratio
        resized = self._resize_preserve_aspect(bordered)
        
        # Normalize to [0, 1] range. Ink is consistently dark (0) and the
        # background is white (1), for both generated samples and canvas crops.
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor if requested
        if return_tensor:
            # Dataset items must be [C, H, W]. DataLoader adds the batch
            # dimension; inference adds it explicitly before model execution.
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            return tensor
        
        return normalized
    
    def _handle_polarity(self, arr: np.ndarray) -> np.ndarray:
        """
        Detect and handle image polarity.
        
        Determines if character is dark on light background or light on dark.
        Ensures consistent output: dark character on light background.
        """
        # The border overwhelmingly belongs to the canvas background. This is
        # more reliable than global pixel counts for thin handwriting.
        border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
        background = float(np.median(border))
        return 255 - arr if background < 128 else arr
    
    def _find_character_region(self, arr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Find bounding box of character region.
        
        Returns:
            Tuple of (min_row, max_row, min_col, max_col)
        """
        # Ink is dark after _handle_polarity. Otsu handles anti-aliased font
        # samples and canvas strokes while the cap avoids selecting white space.
        histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size; weighted_sum = np.dot(np.arange(256), histogram)
        best, best_score, weight_bg, sum_bg = 127, -1.0, 0, 0.0
        for threshold in range(1, 255):
            weight_bg += histogram[threshold]
            if not weight_bg:
                continue
            weight_fg = total - weight_bg
            if not weight_fg:
                break
            sum_bg += threshold * histogram[threshold]
            mean_bg, mean_fg = sum_bg / weight_bg, (weighted_sum - sum_bg) / weight_fg
            score = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if score > best_score:
                best, best_score = threshold, score
        binary = arr <= min(best, 220)
        
        # Find rows and columns with ink
        rows_with_ink = np.any(binary, axis=1)
        cols_with_ink = np.any(binary, axis=0)
        
        if not np.any(rows_with_ink) or not np.any(cols_with_ink):
            # Fallback: use entire image if no ink detected
            return (0, arr.shape[0] - 1, 0, arr.shape[1] - 1)
        
        min_row = np.argmax(rows_with_ink)
        max_row = len(rows_with_ink) - 1 - np.argmax(rows_with_ink[::-1])
        min_col = np.argmax(cols_with_ink)
        max_col = len(cols_with_ink) - 1 - np.argmax(cols_with_ink[::-1])
        
        return (min_row, max_row, min_col, max_col)
    
    def _crop_character(
        self,
        arr: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop the character region from the image."""
        min_row, max_row, min_col, max_col = bbox
        
        # Add small padding to avoid cutting off character edges
        pad = max(1, int((max_row - min_row + 1) * 0.02))
        min_row = max(0, min_row - pad)
        max_row = min(arr.shape[0] - 1, max_row + pad)
        min_col = max(0, min_col - pad)
        max_col = min(arr.shape[1] - 1, max_col + pad)
        
        return arr[min_row:max_row + 1, min_col:max_col + 1]
    
    def _add_border(self, arr: np.ndarray) -> np.ndarray:
        """Add border around cropped character."""
        h, w = arr.shape
        border = int(max(h, w) * self.border_ratio)
        
        # Create new array with border
        new_h = h + 2 * border
        new_w = w + 2 * border
        bordered = np.full((new_h, new_w), self.background_value, dtype=np.uint8)
        
        # Place character in center
        bordered[border:border + h, border:border + w] = arr
        
        return bordered
    
    def _resize_preserve_aspect(self, arr: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while preserving aspect ratio.
        
        The image is scaled to fit within target_size x target_size,
        maintaining aspect ratio, and placed in the center.
        """
        h, w = arr.shape
        
        # Calculate scaling factor to fit within target_size
        scale = self.target_size / max(h, w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Ensure minimum size of 1
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        # Resize using PIL for better quality
        img = Image.fromarray(arr)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized = np.array(img, dtype=np.uint8)
        
        # Create square canvas
        canvas = np.full(
            (self.target_size, self.target_size),
            self.background_value,
            dtype=np.uint8
        )
        
        # Place resized image in center
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas


def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    target_size: int = 64,
    return_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convenience function for preprocessing a single image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Final square image size (default: 64)
        return_tensor: If True, return PyTorch tensor; else return numpy array
    
    Returns:
        Preprocessed image as tensor or array
    """
    preprocessor = HandwritingPreprocessor(target_size=target_size)
    return preprocessor.preprocess(image, return_tensor=return_tensor)


def batch_preprocess(
    images: list,
    target_size: int = 64,
    return_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of images (paths, PIL Images, or numpy arrays)
        target_size: Final square image size (default: 64)
        return_tensor: If True, return PyTorch tensor; else return numpy array
    
    Returns:
        Batch of preprocessed images as tensor or array
    """
    preprocessor = HandwritingPreprocessor(target_size=target_size)
    
    processed = []
    for img in images:
        processed.append(preprocessor.preprocess(img, return_tensor=False))
    
    # Stack into batch
    if return_tensor:
        # Stack arrays and convert to tensor
        batch = np.stack(processed, axis=0)
        # Add channel dimension and convert to tensor
        tensor = torch.from_numpy(batch).unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        return tensor
    
    return np.stack(processed, axis=0)

"""
Handwriting Recognition Dataset Module

This module provides a PyTorch Dataset implementation for handwritten character
images loaded from a CSV manifest.
"""

import csv
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


class HandwritingDataset(Dataset):
    """
    PyTorch Dataset for handwritten character images.
    
    Reads image paths and labels from a CSV manifest file.
    Supports A-Z, a-z, 0-9 characters.
    
    Args:
        csv_path: Path to the CSV manifest file.
        transform: Optional transform to apply to images.
        image_root: Optional root directory to prepend to image paths.
    
    CSV format:
        image_path,label,source,font
        images/A_001.png,A,font,Caveat
        images/B_001.png,B,font,Caveat
        images/a_001.png,a,font,Handlee
        images/0_001.png,0,font,Dancing Script
    """
    
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Any] = None,
        image_root: Optional[str] = None
    ):
        self.csv_path = csv_path
        self.transform = transform
        # Manifest paths are relative to the manifest itself.  Resolving them
        # from the process working directory made training/evaluation silently
        # depend on where a command was launched.
        self.image_root = image_root if image_root is not None else os.path.dirname(os.path.abspath(csv_path))
        
        # Load data from CSV
        self.data: List[Dict[str, str]] = []
        self._load_csv()
        
        # Build label mappings
        self._build_label_mappings()
    
    def _load_csv(self) -> None:
        """Load and validate the CSV manifest."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV manifest not found: {self.csv_path}")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Validate CSV headers
                expected_headers = {'image_path', 'label', 'source', 'font'}
                if not expected_headers.issubset(set(reader.fieldnames or [])):
                    raise ValueError(
                        f"CSV must contain headers: {expected_headers}. "
                        f"Found: {reader.fieldnames}"
                    )
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header row
                    try:
                        image_path = row.get('image_path', '').strip()
                        label = row.get('label', '').strip()
                        source = row.get('source', '').strip()
                        font = row.get('font', '').strip()
                        
                        # Validate required fields
                        if not image_path:
                            raise ValueError(f"Empty image_path at row {row_num}")
                        if not label:
                            raise ValueError(f"Empty label at row {row_num}")
                        if len(label) != 1:
                            raise ValueError(f"Label must be a single character at row {row_num}: '{label}'")
                        
                        # Validate allowed characters
                        if not self._is_valid_character(label):
                            raise ValueError(
                                f"Invalid character '{label}' at row {row_num}. "
                                f"Only A-Z, a-z, and 0-9 are allowed."
                            )
                        
                        # Store data
                        self.data.append({
                            'image_path': image_path,
                            'label': label,
                            'source': source,
                            'font': font
                        })
                        
                    except Exception as e:
                        raise ValueError(f"Error processing row {row_num}: {e}")
        
        except csv.Error as e:
            raise ValueError(f"Malformed CSV file: {e}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data rows found in CSV: {self.csv_path}")
    
    def _is_valid_character(self, char: str) -> bool:
        """Check if character is A-Z, a-z, or 0-9."""
        return (
            ('A' <= char <= 'Z') or 
            ('a' <= char <= 'z') or 
            ('0' <= char <= '9')
        )
    
    def _build_label_mappings(self) -> None:
        """Build deterministic label-to-index and index-to-label mappings."""
        # Get unique labels and sort for determinism
        unique_labels = sorted(set(item['label'] for item in self.data))
        
        # Build mappings
        self.label_to_idx: Dict[str, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.idx_to_label: Dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }
        
        # Store number of classes
        self.num_classes = len(unique_labels)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Tuple of (image_tensor, label_index, original_label)
        
        Raises:
            FileNotFoundError: If the image file is missing.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")
        
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item['image_path'])
        label = item['label']
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}\n"
                f"Check that image_root is correct or paths are absolute."
            )
        
        try:
            image = Image.open(image_path)
            
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply transform if provided
            if self.transform is not None:
                image = self.transform(image)
            
            # Get label index
            label_idx = self.label_to_idx[label]
            
            return image, label_idx, label
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
    
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the label mappings.
        
        Returns:
            Tuple of (label_to_idx, idx_to_label)
        """
        return self.label_to_idx, self.idx_to_label
    
    def get_num_classes(self) -> int:
        """Get the number of unique classes."""
        return self.num_classes


# Convenience function for quick dataset creation
def create_handwriting_dataset(
    csv_path: str,
    transform: Optional[Any] = None,
    image_root: Optional[str] = None
) -> HandwritingDataset:
    """
    Create a HandwritingDataset instance.
    
    Args:
        csv_path: Path to the CSV manifest.
        transform: Optional transform to apply.
        image_root: Optional root directory for images.
    
    Returns:
        HandwritingDataset instance.
    """
    return HandwritingDataset(
        csv_path=csv_path,
        transform=transform,
        image_root=image_root
    )

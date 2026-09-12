"""
Handwriting Recognition Inference Module

This module provides a standalone inference system for the trained handwriting CNN.
It loads a trained checkpoint and provides prediction APIs for single images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from app.recognition.handwriting.preprocessing import HandwritingPreprocessor
from app.recognition.handwriting.model import HandwritingCNN


class HandwritingRecognizer:
    """
    Handwriting character recognition inference system.
    
    Loads a trained checkpoint and provides prediction capabilities for
    handwritten character images.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Optional device to use (cpu or cuda)
        target_size: Target image size (default: 64, must match training)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        target_size: int = 64
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.target_size = target_size
        
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.num_classes = None
        self.checkpoint_info = None
        self.preprocessor = None
        
        # Load model
        self._load_checkpoint()
        self._initialize_preprocessor()
    
    def _load_checkpoint(self) -> None:
        """Load the trained model checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please train the model first using train.py"
            )
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract metadata
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
            'num_classes': checkpoint.get('num_classes', 0),
            'target_size': checkpoint.get('target_size', 64)
        }
        
        # Get label mappings
        self.label_to_idx = checkpoint.get('label_to_idx', {})
        self.idx_to_label = checkpoint.get('idx_to_label', {})
        
        if not self.label_to_idx or not self.idx_to_label:
            raise ValueError(
                "Checkpoint missing label mappings. "
                "Please ensure the checkpoint was created by train.py"
            )
        
        self.num_classes = len(self.label_to_idx)
        
        # Update target size from checkpoint if available
        if 'target_size' in checkpoint:
            self.target_size = checkpoint['target_size']
        
        print(f"Loaded checkpoint from: {self.checkpoint_path}")
        print(f"  Epoch: {self.checkpoint_info['epoch']}")
        print(f"  Best validation accuracy: {self.checkpoint_info['best_val_accuracy']:.2f}%")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Target size: {self.target_size}")
        print(f"  Classes: {sorted(self.label_to_idx.keys())}")
        
        # Create and load model
        self.model = HandwritingCNN(
            num_classes=self.num_classes,
            input_channels=1,
            input_size=self.target_size
        )
        
        # Load state dict
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state dict: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def _initialize_preprocessor(self) -> None:
        """Initialize the preprocessing pipeline."""
        self.preprocessor = HandwritingPreprocessor(
            target_size=self.target_size
        )
    
    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various input types.
        
        Args:
            image: Image path, PIL Image, or numpy array
        
        Returns:
            PIL Image object
        
        Raises:
            ValueError: If image cannot be loaded
        """
        if isinstance(image, str):
            try:
                img = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image from {image}: {e}")
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            try:
                img = Image.fromarray(image)
            except Exception as e:
                raise ValueError(f"Failed to convert numpy array to image: {e}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return img
    
    @torch.no_grad()
    def recognize(
        self,
        image: Union[str, Image.Image, np.ndarray],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Recognize a handwritten character from an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            top_k: Number of top predictions to return (default: 3)
        
        Returns:
            Dictionary containing:
                - label: Predicted character label
                - confidence: Confidence score (0-1)
                - top_k: List of top-k predictions with labels and confidences
                - raw_probs: All class probabilities (optional, for debugging)
        
        Raises:
            ValueError: If image is invalid or preprocessing fails
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check checkpoint loading.")
        
        # Load image
        img = self._load_image(image)
        
        # Preprocess image
        try:
            # Preprocess returns a single [C, H, W] sample; inference adds
            # the batch dimension expected by the CNN.
            processed = self.preprocessor.preprocess(
                img,
                return_tensor=True
            ).unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
        
        # Move to device
        processed = processed.to(self.device)
        
        # Forward pass
        try:
            logits = self.model(processed)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
        
        # Calculate probabilities
        probs = F.softmax(logits, dim=1)
        probs = probs.cpu().numpy().flatten()
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:min(top_k, self.num_classes)]
        top_k_probs = probs[top_k_indices]
        
        # Build top-k predictions list
        top_k_predictions = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            label = self.idx_to_label.get(int(idx), f"class_{idx}")
            top_k_predictions.append({
                'label': label,
                'confidence': float(prob)
            })
        
        # Get best prediction
        best_idx = top_k_indices[0]
        best_label = self.idx_to_label.get(int(best_idx), f"class_{best_idx}")
        best_confidence = float(top_k_probs[0])
        
        # Build result
        result = {
            'label': best_label,
            'confidence': best_confidence,
            'top_k': top_k_predictions,
            'raw_probs': probs.tolist()  # Include raw probabilities for debugging
        }
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recognize multiple images in batch.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            top_k: Number of top predictions to return
        
        Returns:
            List of recognition results
        """
        results = []
        for image in images:
            try:
                result = self.recognize(image, top_k=top_k)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'label': None,
                    'confidence': 0.0,
                    'top_k': []
                })
        return results
    
    def get_class_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Get the label mappings used by the model.
        
        Returns:
            Tuple of (label_to_idx, idx_to_label)
        """
        return self.label_to_idx, self.idx_to_label
    
    def get_num_classes(self) -> int:
        """Get the number of classes the model supports."""
        return self.num_classes
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the loaded checkpoint."""
        return self.checkpoint_info


def format_prediction_result(result: Dict[str, Any]) -> str:
    """
    Format recognition result for display.
    
    Args:
        result: Recognition result from HandwritingRecognizer.recognize()
    
    Returns:
        Formatted string for display
    """
    if 'error' in result:
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append(f"\n{'='*50}")
    lines.append(f"PREDICTION RESULT")
    lines.append(f"{'='*50}")
    lines.append(f"Predicted Character: {result['label']}")
    lines.append(f"Confidence: {result['confidence']*100:.2f}%")
    
    if result['top_k'] and len(result['top_k']) > 1:
        lines.append(f"\nTop {len(result['top_k'])} Predictions:")
        for i, pred in enumerate(result['top_k'], 1):
            lines.append(f"  {i}. {pred['label']} — {pred['confidence']*100:.2f}%")
    
    lines.append(f"{'='*50}")
    return '\n'.join(lines)


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Handwriting recognition inference")
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="app/data/models/handwriting/checkpoint_best.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize recognizer
        print("Loading handwriting recognizer...")
        recognizer = HandwritingRecognizer(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        # Recognize image
        print(f"\nRecognizing image: {args.image}")
        result = recognizer.recognize(args.image, top_k=args.top_k)
        
        # Display result
        print(format_prediction_result(result))
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease ensure:")
        print("  1. The model is trained first (run train.py)")
        print("  2. The checkpoint path is correct")
        print("  3. The image path exists")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
